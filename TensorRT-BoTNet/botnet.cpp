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

static const float kMean[3] = {0.485, 0.456, 0.406};
static const float kStdDev[3] = {0.229, 0.224, 0.225};


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

IShuffleLayer* mhsaLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor&input, int outch, std::string lname, int heads=4){
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
  
    IConvolutionLayer* q = network->addConvolutionNd(input, outch, DimsHW{1,1}, weightMap[lname + "conv2.0.query.weight"], weightMap[lname + "conv2.0.query.bias"]);
    assert(q);
    const Dims idims = q->getOutput(0)->getDimensions();
    const int width = idims.d[1];
    const int height = idims.d[2];
  
    IShuffleLayer* q_view = network->addShuffle(*q->getOutput(0));
    assert(q_view);
    q_view->setReshapeDimensions( Dims3{ heads, outch / heads, width*height});

    IConvolutionLayer* k = network->addConvolutionNd(input, outch, DimsHW{1,1}, weightMap[lname + "conv2.0.key.weight"], weightMap[lname + "conv2.0.key.bias"]);
    assert(k);

    IShuffleLayer* k_view = network->addShuffle(*k->getOutput(0));
    assert(k_view);
    k_view->setReshapeDimensions( Dims3{ heads, outch / heads, width*height});

    IConvolutionLayer* v = network->addConvolutionNd(input, outch, DimsHW{1,1}, weightMap[lname + "conv2.0.value.weight"], weightMap[lname + "conv2.0.value.bias"]);
    assert(v);
    IShuffleLayer* v_view = network->addShuffle(*v->getOutput(0));
    assert(v_view);
    v_view->setReshapeDimensions( Dims3{ heads, outch / heads, width*height});

    auto content_content = network->addMatrixMultiply(*q_view->getOutput(0), MatrixOperation::kTRANSPOSE, *k_view->getOutput(0), MatrixOperation::kNONE);
    assert(content_content );

    auto rel_h = network->addConstant(Dims5{1, 4, outch / heads, 1, height}, weightMap[lname + "conv2.0.rel_h"]);
    auto rel_w = network->addConstant(Dims5{1, 4, outch / heads, width, 1}, weightMap[lname + "conv2.0.rel_w"]);

    IElementWiseLayer* content_position= network->addElementWise(*rel_h->getOutput(0), *rel_w->getOutput(0), ElementWiseOperation::kSUM);
    assert(content_position);

    IShuffleLayer* content_position_view = network->addShuffle(*content_position->getOutput(0));
    assert(content_position_view);
    content_position_view->setReshapeDimensions( Dims3{ heads, outch / heads, width*height});

    auto content_position22 = network->addMatrixMultiply(*content_position_view->getOutput(0),MatrixOperation::kTRANSPOSE, *q_view->getOutput(0),MatrixOperation::kNONE);
    assert(content_position22);

    IElementWiseLayer* energy = network->addElementWise(*content_content->getOutput(0), *content_position22->getOutput(0),ElementWiseOperation::kSUM);
    assert(energy);

    IShuffleLayer* energy_view= network->addShuffle(*energy->getOutput(0));
    assert(energy_view);
    energy_view->setSecondTranspose(Permutation{2, 1, 0});//0,1,2

    auto attention = network->addSoftMax(*energy_view->getOutput(0));
    IShuffleLayer* attention_view= network->addShuffle(*attention->getOutput(0));
    assert(attention_view);
    attention_view->setSecondTranspose(Permutation{2, 1, 0}); 
    auto out = network->addMatrixMultiply(*v_view->getOutput(0),MatrixOperation::kNONE, *attention_view->getOutput(0),MatrixOperation::kTRANSPOSE);
    assert(out);
    std::cout<< "out" <<std::endl;
    
    IShuffleLayer* out_view = network->addShuffle(*out->getOutput(0));
    assert(out_view);
    out_view->setReshapeDimensions( Dims3{outch, width, height});

    const Dims idims1 = out_view->getOutput(0)->getDimensions();
 
    std::cout<< idims1.d[0]<<std::endl;
    std::cout<< idims1.d[1]<<std::endl;
    std::cout<< idims1.d[2]<<std::endl;
    std::cout<< idims1.d[3]<<std::endl;
    return out_view;

}
IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride,  std::string lname, bool mhsa =false, int heads=4 ) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    std::cout<< "bottleneck_conv1" <<std::endl;

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    const Dims idims11 = relu1->getOutput(0)->getDimensions();

    std::cout<<"relu1"<< idims11.d[0]<<std::endl;
    std::cout<< idims11.d[1]<<std::endl;
    std::cout<< idims11.d[2]<<std::endl;
    std::cout<< idims11.d[3]<<std::endl;
    IScaleLayer* bn2;
    if (mhsa==false){

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        assert(conv2);
        conv2->setStrideNd(DimsHW{stride, stride});
        conv2->setPaddingNd(DimsHW{1, 1});       
        bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);
        std:: cout<< "print" << std::endl;
    }else{
        std::cout<< "mhsa" <<std::endl;
        IShuffleLayer*MHSA = mhsaLayer(network, weightMap, *relu1->getOutput(0), outch, lname, 4);
        if (stride ==2){
            IPoolingLayer* pool = network->addPoolingNd(*MHSA->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
            assert(pool);
            pool->setStrideNd(DimsHW{2, 2});
            pool->setPaddingNd(DimsHW{0, 0});

            std::cout<< "strid==2" <<std::endl;


            bn2 = addBatchNorm2d(network, weightMap, *pool->getOutput(0), lname + "bn2", 1e-5);
            std:: cout<< "printmhas" << std::endl;
        } else{
            bn2 = addBatchNorm2d(network, weightMap, *MHSA->getOutput(0), lname + "bn2", 1e-5);
        }
       
    }

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);
    std::cout<<"bn3"<<std::endl;


    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "shortcut.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});
        std::cout<<"conv4"<<std::endl;
        const Dims idims3 = conv4->getOutput(0)->getDimensions();
        std::cout<<"conv4"<< idims3.d[0]<<std::endl;
        std::cout<< idims3.d[1]<<std::endl;
        std::cout<< idims3.d[2]<<std::endl;
        std::cout<< idims3.d[3]<<std::endl;


        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "shortcut.1", 1e-5);
        std::cout<<"bn4"<<std::endl;
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        std::cout<<"ew1"<<std::endl;
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    std::cout<<"relu3"<<std::endl;
    return relu3;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{6, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../BotNet.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, "layer1.2.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2,  "layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,  "layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,  "layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,  "layer2.3.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2,  "layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,  "layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,  "layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,  "layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,  "layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,  "layer3.5.");
    x->getOutput(0)->setName(X_BLOB_NAME);
    network->markOutput(*x->getOutput(0)); //标记输出的变量
        
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2, "layer4.0.",true);
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, "layer4.1.",true);
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1,  "layer4.2.",true);
   
    

    IPoolingLayer* pool2 = network->addPoolingNd(*x->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStrideNd(DimsHW{1, 1});

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 7, weightMap["fc.1.weight"], weightMap["fc.1.bias"]);
    assert(fc1);

    const Dims idimsfc = fc1->getOutput(0)->getDimensions();

    // std::cout<<"relu1"<< idimsf<<std::endl;
    std::cout<<"relufc:"<< idimsfc.d[0]<<std::endl;
    std::cout<< idimsfc.d[1]<<std::endl;
    std::cout<< idimsfc.d[2]<<std::endl;
    std::cout<< idimsfc.d[3]<<std::endl;


    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*fc1->getOutput(0));

    // auto prob = network->addSoftMax(*fc1->getOutput(0));
    // prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    // std::cout << "set name out" << std::endl;
    // network->markOutput(*prob->getOutput(0)); //标记输出的变量

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

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
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
        std::cerr << "./botnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./botnet -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("botnet.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("botnet.engine", std::ios::binary);
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
    infile.open("/home/jgh/GrainDataSet/train_test_txt/test1.txt");
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

            data[i + 3*224*224] =((float)img2.at<cv::Vec3b>(i)[0]/255.0f - 0.485) /0.229;
            data[i + 4*224*224] = ((float)img2.at<cv::Vec3b>(i)[1]/255.0f - 0.456) / 0.224;
            data[i + 5*224*224] = ((float)img2.at<cv::Vec3b>(i)[2]/255.0f - 0.406) / 0.225;
        }

        for (int i = 0; i < 2; i++) {
            auto start = std::chrono::system_clock::now();

            doInference(*context, data, prob, 1);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        }

        // Print histogram of the output distribution
        std::cout << "\nOutput:\n\n";
        std::ofstream outfile("../botnet_tensorrt_softmax_test1.txt", std::ios::app);
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
