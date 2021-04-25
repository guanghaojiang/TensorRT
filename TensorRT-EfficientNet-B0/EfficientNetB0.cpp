#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "dirent.h"
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


#define BATCH_SIZE 1
// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 7;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* hSwish(INetworkDefinition *network, ITensor& input, std::string name) {
    auto hsig = network->addActivation(input, ActivationType::kSIGMOID);
    assert(hsig);
    ILayer* hsw = network->addElementWise(input, *hsig->getOutput(0),ElementWiseOperation::kPROD);
    assert(hsw);
    return hsw;
}


ILayer* expandConvBnHswish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "_expand_conv.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn0 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn0", 1e-3);
    ILayer* hsw = hSwish(network, *bn0->getOutput(0), lname+"expand");
    assert(hsw);
    return hsw;
}
    //tf padding:same 非对称填充，左小右大 
int paddingSize(int inSize, int ksize, int stridSize) {

    int mode = inSize % stridSize;
    if (mode == 0)
        return std::max(ksize - stridSize, 0);
    return std::max(ksize - (inSize % stridSize), 0);
}

ILayer* depthWiseConvBnHswish(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s,  std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "_depthwise_conv.weight"], emptywts);
    assert(conv1);
    Dims d = input.getDimensions();
    std:: cout<< "height::"<< d.d[1]<<std::endl;
    conv1->setStrideNd(DimsHW{s, s});

    // tf padding "same"
    int padSize = paddingSize(d.d[1], ksize, s);
    int postPadding = ceil(padSize / 2.0);
    int prePadding = padSize - postPadding;
    if (prePadding > 0)
        conv1->setPrePadding(DimsHW{prePadding, prePadding});
    if (postPadding > 0)
        conv1->setPostPadding(DimsHW{postPadding, postPadding});

    conv1->setNbGroups(outch);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn1", 1e-3);
    ILayer* hsw = hSwish(network, *bn1->getOutput(0), lname+"depth");
    assert(hsw);
    return hsw;
}

ILayer* seLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int w, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    
    int h = w;
    IPoolingLayer* l1 = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW(w, h));
    assert(l1);
    l1->setStrideNd(DimsHW{w, h});
    IConvolutionLayer* l2 = network->addConvolutionNd(*l1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "_se_reduce.weight"], weightMap[lname + "_se_reduce.bias"]);
    assert(l2);
    ILayer* hsw = hSwish(network, *l2->getOutput(0), lname+"se");
    assert(hsw);
    IConvolutionLayer* l4 = network->addConvolutionNd(*hsw->getOutput(0), inch, DimsHW{1, 1}, weightMap[lname + "_se_expand.weight"], weightMap[lname + "_se_expand.bias"]);
    assert(l4);
    auto hsig = network->addActivation(*l4->getOutput(0), ActivationType::kSIGMOID);
    assert(hsig);
    ILayer* se = network->addElementWise(input, *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(se);
    return se;
}

ILayer* projectConv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "_project_conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "_bn2", 1e-3);
    assert(bn2);

    return bn2;
}

  // s:表示strid； se_re: 表示seLayer 的通道降低率； pro_re： 表示projectConv 输出通道的降低率； w : 表示特征图的宽度; expand 表示特征通道是否扩展
ILayer* MBConvBlock(INetworkDefinition * network, std::map<std::string, Weights>&weightMap, ITensor&input, int outch, int ksize, int s,  int se_re, int pro_out, int w ,bool expand, std::string lname ){

    Dims d_input = input.getDimensions();
    ILayer* x2;
    if (expand){
        ILayer* x1 = expandConvBnHswish(network, weightMap, input, outch, lname);
        x2 = depthWiseConvBnHswish(network, weightMap, *x1->getOutput(0), outch, ksize, s,  lname);
    }else{
        x2 = depthWiseConvBnHswish(network, weightMap, input, outch, ksize, s,  lname);
    }
  
    ILayer* x3 = seLayer(network, weightMap, *x2->getOutput(0), outch, outch / se_re, w, lname);

    ILayer* x4 = projectConv(network, weightMap, *x3->getOutput(0), pro_out, lname);
    ILayer* ew1;
    if (d_input.d[0]== pro_out){
        ew1 = network->addElementWise(input, *x4->getOutput(0), ElementWiseOperation::kSUM);
        assert(ew1);

        return ew1;
    } else{
        return x4;
    }   
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{6, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../EfficientNet-B0.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    //stem  32 x 112 x 112
    IConvolutionLayer* conv_stem = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, weightMap["_conv_stem.weight"], emptywts);
    assert(conv_stem);
    // Dims da = data.getDimensions();
    conv_stem->setStrideNd(DimsHW{2, 2});
    
    // tf padding "same"
    int padSize = paddingSize(224, 3, 2);
    int postPadding = ceil(padSize / 2.0);
    int prePadding = padSize - postPadding;
    if (prePadding > 0)
        conv_stem->setPrePadding(DimsHW{prePadding, prePadding});
    if (postPadding > 0)
        conv_stem->setPostPadding(DimsHW{postPadding, postPadding});

    IScaleLayer* _bn0 = addBatchNorm2d(network, weightMap, *conv_stem->getOutput(0), "_bn0", 1e-3);

    // Add activation layer using the swish algorithm.
    ILayer* sw1 = hSwish(network, *_bn0->getOutput(0), "stem");
    assert(sw1);
   

    // Blocks 
    // MBConvBlock(INetworkDefinition * network, std::map<std::string, Weights>&weightMap, ITensor&input,  float outch, int ksize, int s, int se_re, int pro_re, int w ,bool expand, std::string lname )
    // s:表示strid； se_re: 表示seLayer 的通道降低率； pro_re： 表示projectConv 输出通道的降低率； w : 表示特征图的宽度; expand 表示特征通道是否扩展
  
    auto x0 = MBConvBlock(network, weightMap, *sw1->getOutput(0), 32, 3, 1,  4, 16, 112, false,  "_blocks.0.");

    auto x1 = MBConvBlock(network, weightMap, *x0->getOutput(0), 16, 3, 1, 4, 16, 112, false,  "_blocks.1.");

    auto x2 = MBConvBlock(network, weightMap, *x1->getOutput(0), 96, 3,  2, 24, 24, 56, true, "_blocks.2.");

    auto x3 = MBConvBlock(network, weightMap, *x2->getOutput(0), 144, 3, 1,  24, 24, 56, true, "_blocks.3.");
    auto x4 = MBConvBlock(network, weightMap, *x3->getOutput(0), 144, 3, 1,  24, 24, 56, true, "_blocks.4.");
    auto x5 = MBConvBlock(network, weightMap, *x4->getOutput(0), 144, 5, 2,  24, 48, 28, true, "_blocks.5.");

    auto x6 = MBConvBlock(network, weightMap, *x5->getOutput(0), 288, 5, 1,  24, 48, 28, true, "_blocks.6.");
    auto x7 = MBConvBlock(network, weightMap, *x6->getOutput(0), 288, 5, 1,  24, 48, 28, true, "_blocks.7.");
    auto x8 = MBConvBlock(network, weightMap, *x7->getOutput(0), 288, 3, 2, 24, 88, 14, true, "_blocks.8.");

    auto x9 = MBConvBlock(network, weightMap, *x8->getOutput(0), 528, 3, 1,  24, 88, 14, true, "_blocks.9.");
    auto x10 = MBConvBlock(network, weightMap, *x9->getOutput(0), 528, 3, 1,  24, 88, 14, true, "_blocks.10.");  
    auto x11 = MBConvBlock(network, weightMap, *x10->getOutput(0), 528, 3, 1,  24, 88, 14, true, "_blocks.11.");
    auto x12 = MBConvBlock(network, weightMap, *x11->getOutput(0), 528, 5, 1,  24, 120, 14, true, "_blocks.12.");

    auto x13 = MBConvBlock(network, weightMap, *x12->getOutput(0), 720, 5, 1,  24, 120, 14, true, "_blocks.13.");
    auto x14 = MBConvBlock(network, weightMap, *x13->getOutput(0), 720, 5, 1,  24, 120, 14, true, "_blocks.14.");
    auto x15 = MBConvBlock(network, weightMap, *x14->getOutput(0), 720, 5, 1,  24, 120, 14, true, "_blocks.15.");
    auto x16 = MBConvBlock(network, weightMap, *x15->getOutput(0), 720, 5, 2, 24, 208, 7, true, "_blocks.16.");

    auto x17 = MBConvBlock(network, weightMap, *x16->getOutput(0), 1248, 5, 1,  24, 208, 7, true, "_blocks.17.");
    auto x18 = MBConvBlock(network, weightMap, *x17->getOutput(0), 1248, 5, 1,  24, 208, 7, true, "_blocks.18.");
    auto x19 = MBConvBlock(network, weightMap, *x18->getOutput(0), 1248, 5, 1,  24, 208, 7, true, "_blocks.19.");
    auto x20 = MBConvBlock(network, weightMap, *x19->getOutput(0), 1248, 5, 1,  24, 208, 7, true, "_blocks.20.");
    auto x21 = MBConvBlock(network, weightMap, *x20->getOutput(0), 1248, 3, 1,  24, 352, 7, true, "_blocks.21.");
    auto x22 = MBConvBlock(network, weightMap, *x21->getOutput(0), 2112, 3, 1,  24, 352, 7, true, "_blocks.22.");

    //conv_head

    IConvolutionLayer* conv_head = network->addConvolutionNd(*x22->getOutput(0), 1408, DimsHW{1, 1}, weightMap["_conv_head.weight"], emptywts);
    assert(conv_head);
    conv_head->setStrideNd(DimsHW{1, 1});


    IScaleLayer* _bn1 = addBatchNorm2d(network, weightMap, *conv_head->getOutput(0), "_bn1", 1e-3);

    // Add activation layer using the swish algorithm.
    ILayer* sw2 = hSwish(network, *_bn1->getOutput(0), "stem");
    assert(sw2);


    IPoolingLayer* pool2 = network->addPoolingNd(*sw2->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 7, weightMap["_fc.weight"], weightMap["_fc.bias"]);
    assert(fc1);

    const Dims idimsfc = fc1->getOutput(0)->getDimensions();

    // std::cout<<"relu1"<< idimsf<<std::endl;
    std::cout<<"relufc:"<< idimsfc.d[0]<<std::endl;
    std::cout<< idimsfc.d[1]<<std::endl;
    std::cout<< idimsfc.d[2]<<std::endl;
    std::cout<< idimsfc.d[3]<<std::endl;


    // fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    // std::cout << "set name out" << std::endl;
    // network->markOutput(*fc1->getOutput(0));

    auto prob = network->addSoftMax(*fc1->getOutput(0));
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*prob->getOutput(0)); //标记输出的变量

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}


void doInference(IExecutionContext& context, float* input, float* output,  int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 6 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));


    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 6 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./efficientnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./efficientnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("efficientnet-b0.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("efficientnet-b0.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    static float data[6 * INPUT_H * INPUT_W];
    // for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //     data[i] = 1.0;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    //-------------------Preprocessed data -------------------------------------------------------------

    std::vector<std::string> path,label;
    std::fstream infile; 
    infile.open("/home/jgh/GrainDataSet/train_test_txt/test.txt");
    assert(infile.is_open());

    std::string s ,s1,s2;
    while(getline(infile, s))
    {
        std::istringstream is(s);   //将字符串按空格分开
        is >> s1 >>s2;
        path.push_back(s1);
        label.push_back(s2);
    }
    infile.close();
    for (int i=0; i<path.size();i++)
    {
        cv::Mat img1,img2;
        cv::Mat image = cv::imread(path[i]);
        auto width = image.cols;
        auto height = image.rows;
        auto strid = width / 3 ;
        
        cv::Rect roi1(0,0,strid,height);
        cv::Rect roi2(strid,0,strid,height);
        cv::Mat image_roi1 = image(roi1);
        cv::Mat image_roi2 = image(roi2);

        cv::resize(image_roi1, img1, cv::Size(224,224),0,0, cv::INTER_CUBIC);
        cv::resize(image_roi2, img2, cv::Size(224,224),0,0, cv::INTER_CUBIC);
        
        static float data[6* 224 * 224];
        for (int i = 0; i< 224 *224;i++){
            data[i]=((float)img1.at<cv::Vec3b>(i)[0]/255.0f - 0.485) /0.229;
            data[i + 224*224] = ((float)img1.at<cv::Vec3b>(i)[1]/255.0f - 0.456) / 0.224;
            data[i + 2*224*224] = ((float)img1.at<cv::Vec3b>(i)[2]/255.0f - 0.406) / 0.225;

            data[i + 3*224*224] =((float)img2.at<cv::Vec3b>(i)[0]/255.0f - 0.485) / 0.229;
            data[i + 4*224*224] = ((float)img2.at<cv::Vec3b>(i)[1]/255.0f - 0.456) / 0.224;
            data[i + 5*224*224] = ((float)img2.at<cv::Vec3b>(i)[2]/255.0f - 0.406) / 0.225;
        }

        for (int i = 0; i < 2; i++) {
            auto start = std::chrono::system_clock::now();

            doInference(*context, data, prob,  1);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        }

        std::cout << "\nOutput:\n\n";
        std::ofstream outfile("../efficientnet_tensorrt_softmax1.txt", std::ios::app);
        assert(outfile.is_open());
        for (unsigned int i = 0; i < 7; i++)
        {
            outfile<<prob[i]<<" ";
            std::cout << prob[i] << ", ";
        }
        outfile<< "\n";
        outfile.close();
        std::cout << std::endl;
    }
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
