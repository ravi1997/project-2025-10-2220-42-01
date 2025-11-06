// main.cpp — usage/demo for stdlib-only DNN
#include "dnn.hpp"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>

using namespace dnn;

struct Args {
    std::string demo{"xor"};  // xor | 3class
    unsigned int seed{0};     // 0 => random_device
};

static std::optional<unsigned> to_uint(std::string_view s){
    unsigned v{}; auto r = std::from_chars(s.data(), s.data()+s.size(), v);
    if (r.ec==std::errc()) return v; return std::nullopt;
}

static Args parse_args(int argc, char** argv){
    Args a{};
    for (int i=1;i<argc;++i){
        std::string_view t = argv[i];
        auto next = [&](int i)->std::string_view{
            if (i+1>=argc){ std::cerr << "missing value for " << t << "\n"; std::exit(2); }
            return argv[i+1];
        };
        if (t=="--demo"){ a.demo = std::string(next(i)); ++i; }
        else if (t=="--seed"){ if (auto u=to_uint(next(i))) a.seed=*u; else { std::cerr<<"bad --seed\n"; std::exit(2);} ++i; }
        else if (t=="--help" || t=="-h"){
            std::cout << "Usage: dnn --demo xor|3class [--seed N]\n";
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << t << "\n"; std::exit(2);
        }
    }
    return a;
}

static void demo_xor(std::mt19937& rng){
    std::cout << "\n=== XOR demo (2 -> 8 -> 1, sigmoid, MSE) ===\n";
    Matrix X({4,2}); X.data()[0]=0; X.data()[1]=0; X.data()[2]=0; X.data()[3]=1; X.data()[4]=1; X.data()[5]=0; X.data()[6]=1; X.data()[7]=1;
    Matrix y({4,1}); y.data()[0]=0; y.data()[1]=1; y.data()[2]=1; y.data()[3]=0;

    // Create model using the correct API
    Config config;
    Model net(config);
    
    // Add layers using the correct API
    net.add(std::make_unique<Dense>(2, 8, Activation::ReLU));
    net.add(std::make_unique<Dense>(8, 1, Activation::Sigmoid));

    // Compile model with optimizer
    auto optimizer = std::make_unique<SGD>(0.1, 0.9, 1e-4); // lr, momentum, weight_decay
    net.compile(std::move(optimizer));

    // Train the model
    net.fit(X, y, 2000, LossFunction::MSE, rng, 0.0, true);

    // Make predictions
    Matrix ypred = net.predict(X);
    for (std::size_t i=0;i<4;++i)
        std::cout << int(X.data()[i*X.shape()[1]]) << " xor " << int(X.data()[i*X.shape()[1]+1]) << " -> " << ypred.data()[i*ypred.shape()[1]] << "\n";
}

static void demo_3class(std::mt19937& rng){
    std::cout << "\n=== 3-class toy demo (2 -> 16 -> 3, softmax + CE) ===\n";
    std::normal_distribution<double> N(0.0, 0.40);
    const int per_class=60, C=3;
    Matrix X({static_cast<std::size_t>(per_class*C), 2});
    std::vector<int> labels; labels.reserve(static_cast<std::size_t>(per_class*C));

    auto put = [&](int cls, double cx, double cy){
        for (int i=0;i<per_class;++i){
            std::size_t r=labels.size();
            X.data()[r*X.shape()[1]]=cx+N(rng); X.data()[r*X.shape()[1]+1]=cy+N(rng);
            labels.push_back(cls);
        }
    };
    put(0, -1.5,  0.0);
    put(1,  1.5,  0.0);
    put(2,  0.0,  1.5);

    Matrix y = one_hot(labels, C);

    // Create model using the correct API
    Config config;
    Model net(config);
    
    // Add layers using the correct API
    net.add(std::make_unique<Dense>(2, 16, Activation::ReLU));
    net.add(std::make_unique<Dense>(16, 3, Activation::Softmax));

    // Compile model with optimizer
    auto optimizer = std::make_unique<SGD>(0.1, 0.9, 5e-4); // lr, momentum, weight_decay
    net.compile(std::move(optimizer));

    // Train the model
    net.fit(X, y, 450, LossFunction::CrossEntropy, rng, 0.2, true);

    // Evaluate accuracy
    Matrix predictions = net.predict(X);
    double acc = accuracy(predictions, labels);
    std::cout << "final acc: " << acc << "\n";
}

int main(int argc, char** argv){
    try {
        auto args = parse_args(argc, argv);
        std::random_device rd;
        unsigned int seed = args.seed ? args.seed : rd();
        std::mt19937 rng(seed);

        if (args.demo == "xor")      demo_xor(rng);
        else if (args.demo=="3class") demo_3class(rng);
        else { std::cerr << "unknown --demo (use xor|3class)\n"; return 2; }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}