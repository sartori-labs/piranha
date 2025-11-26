
#include "unitTests.h" 
extern Profiler comm_profiler;

std::default_random_engine generator(0xffa0);

template<typename T>
struct EvalTest: public testing::Test {};

void random_vector(std::vector<double> &v, int size) {

    std::normal_distribution<double> distribution(0.0, 10.0);

    v.clear();
    v.resize(size);

    for (int i = 0; i < v.size(); i++) {
        v[i] = distribution(generator);
    }
}

TEST(EvalTest, MatMul_2PC_Profiling) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {1, 10, 30, 50, 100, 300};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n * n);
        TPC<uint64_t> a(n*n);
        a.setPublic(rnd_vals);

        random_vector(rnd_vals, n * n);
        TPC<uint64_t> b(n*n);
        b.setPublic(rnd_vals);

        TPC<uint64_t> c(n*n);

        Profiler profiler;
        profiler.start();

        matmul(a, b, c, n, n, n, false, false, false, (uint64_t)FLOAT_PRECISION);

        profiler.accumulate("matmul");
        // printf("I am here\n");
        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - matmul (N=%d) - %f sec.\n", n, profiler.get_elapsed("matmul") / 1000.0);
    }
}

TEST(EvalTest, MatMul_3PC_Profiling) {

    if (partyNum >= 3) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {1, 10, 30, 50, 100, 300};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n * n);
        RSS<uint64_t> a(n*n);
        a.setPublic(rnd_vals);

        random_vector(rnd_vals, n * n);
        RSS<uint64_t> b(n*n);
        b.setPublic(rnd_vals);

        RSS<uint64_t> c(n*n);

        Profiler profiler;
        profiler.start();

        matmul(a, b, c, n, n, n, false, false, false, (uint64_t)FLOAT_PRECISION);

        profiler.accumulate("matmul");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("3PC - matmul (N=%d) - %f sec.\n", n, profiler.get_elapsed("matmul") / 1000.0);
    }
}

TEST(EvalTest, MatMul_4PC_Profiling) {

    if (partyNum >= 4) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {1, 10, 30, 50, 100, 300};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n * n);
        FPC<uint64_t> a(n*n);
        a.setPublic(rnd_vals);

        random_vector(rnd_vals, n * n);
        FPC<uint64_t> b(n*n);
        b.setPublic(rnd_vals);

        FPC<uint64_t> c(n*n);

        Profiler profiler;
        profiler.start();

        matmul(a, b, c, n, n, n, false, false, false, (uint64_t)FLOAT_PRECISION);

        profiler.accumulate("matmul");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("4PC - matmul (N=%d) - %f sec.\n", n, profiler.get_elapsed("matmul") / 1000.0);
    }
}

TEST(EvalTest, Conv_2PC_Profiling) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<std::tuple<int, int, int, int, int> > dims = {
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(12, 20, 50, 3, 10),
        std::make_tuple(32, 3, 50, 7, 24),
        std::make_tuple(64, 3, 32, 5, 60),
        std::make_tuple(224, 3, 64, 5, 220)
    };
    for (int i = 0; i < dims.size(); i++) {

        auto dim = dims[i];

        int im_size = std::get<0>(dim);
        int din = std::get<1>(dim);
        int dout = std::get<2>(dim);
        int f_size = std::get<3>(dim);
        int out_size = std::get<4>(dim);

        int N = 1;

        int a_size = N * din * im_size * im_size;
        random_vector(rnd_vals, a_size);
        TPC<uint64_t> a(a_size);
        a.setPublic(rnd_vals);

        int b_size = din * dout * f_size * f_size;
        random_vector(rnd_vals, b_size);
        TPC<uint64_t> b(b_size);
        b.setPublic(rnd_vals);

        TPC<uint64_t> c(N * dout * out_size * out_size);

        Profiler profiler;
        profiler.start();

        convolution(a, b, c, cutlass::conv::Operator::kFprop, N, im_size, im_size, f_size, din, dout, 1, 0, FLOAT_PRECISION);

        profiler.accumulate("conv");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - conv (N=1, Iw/h=%d, Din=%d, Dout=%d, f=%d) - %f sec.\n", im_size, din, dout, f_size, profiler.get_elapsed("conv") / 1000.0);
    }
}

TEST(EvalTest, Conv_3PC_Profiling) {

    if (partyNum >= 3) return;

    std::vector<double> rnd_vals;

    std::vector<std::tuple<int, int, int, int, int> > dims = {
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(12, 20, 50, 3, 10),
        std::make_tuple(32, 3, 50, 7, 24),
        std::make_tuple(64, 3, 32, 5, 60),
        std::make_tuple(224, 3, 64, 5, 220)
    };
    for (int i = 0; i < dims.size(); i++) {

        auto dim = dims[i];

        int im_size = std::get<0>(dim);
        int din = std::get<1>(dim);
        int dout = std::get<2>(dim);
        int f_size = std::get<3>(dim);
        int out_size = std::get<4>(dim);

        int N = 1;

        int a_size = N * din * im_size * im_size;
        random_vector(rnd_vals, a_size);
        RSS<uint64_t> a(a_size);
        a.setPublic(rnd_vals);

        int b_size = din * dout * f_size * f_size;
        random_vector(rnd_vals, b_size);
        RSS<uint64_t> b(b_size);
        b.setPublic(rnd_vals);

        RSS<uint64_t> c(N * dout * out_size * out_size);

        Profiler profiler;
        profiler.start();

        convolution(a, b, c, cutlass::conv::Operator::kFprop, N, im_size, im_size, f_size, din, dout, 1, 0, FLOAT_PRECISION);

        profiler.accumulate("conv");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("3PC - conv (N=1, Iw/h=%d, Din=%d, Dout=%d, f=%d) - %f sec.\n", im_size, din, dout, f_size, profiler.get_elapsed("conv") / 1000.0);
    }
}

TEST(EvalTest, Conv_4PC_Profiling) {

    if (partyNum >= 4) return;

    std::vector<double> rnd_vals;

    std::vector<std::tuple<int, int, int, int, int> > dims = {
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(28, 1, 16, 5, 24),
        std::make_tuple(12, 20, 50, 3, 10),
        std::make_tuple(32, 3, 50, 7, 24),
        std::make_tuple(64, 3, 32, 5, 60),
        std::make_tuple(224, 3, 64, 5, 220)
    };
    for (int i = 0; i < dims.size(); i++) {

        auto dim = dims[i];

        int im_size = std::get<0>(dim);
        int din = std::get<1>(dim);
        int dout = std::get<2>(dim);
        int f_size = std::get<3>(dim);
        int out_size = std::get<4>(dim);

        int N = 1;

        int a_size = N * din * im_size * im_size;
        random_vector(rnd_vals, a_size);
        FPC<uint64_t> a(a_size);
        a.setPublic(rnd_vals);

        int b_size = din * dout * f_size * f_size;
        random_vector(rnd_vals, b_size);
        FPC<uint64_t> b(b_size);
        b.setPublic(rnd_vals);

        FPC<uint64_t> c(N * dout * out_size * out_size);

        Profiler profiler;
        profiler.start();

        convolution(a, b, c, cutlass::conv::Operator::kFprop, N, im_size, im_size, f_size, din, dout, 1, 0, FLOAT_PRECISION);

        profiler.accumulate("conv");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("4PC - conv (N=1, Iw/h=%d, Din=%d, Dout=%d, f=%d) - %f sec.\n", im_size, din, dout, f_size, profiler.get_elapsed("conv") / 1000.0);
    }
}

TEST(EvalTest, ReLU_2PC_Profiling) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {32, 512, 8192, 16384};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        TPC<uint64_t> a(n);
        a.setPublic(rnd_vals);

        TPC<uint64_t> c(n);
        TPC<uint8_t> dc(n);

        Profiler profiler;
        profiler.start();

        ReLU(a, c, dc);

        profiler.accumulate("relu");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - relu (N=%d) - %f sec.\n", n, profiler.get_elapsed("relu") / 1000.0);
    }
}

TEST(EvalTest, ReLU_3PC_Profiling) {

    if (partyNum >= 3) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {1, 10, 100, 1000, 10000, 100000};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        RSS<uint64_t> a(n);
        a.setPublic(rnd_vals);

        RSS<uint64_t> c(n);
        RSS<uint8_t> dc(n);

        Profiler profiler;
        profiler.start();

        ReLU(a, c, dc);

        profiler.accumulate("relu");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("3PC - relu (N=%d) - %f sec.\n", n, profiler.get_elapsed("relu") / 1000.0);
    }
}

TEST(EvalTest, ReLU_4PC_Profiling) {

    if (partyNum >= 4) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {1, 10, 100, 1000, 10000, 100000};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        FPC<uint64_t> a(n);
        a.setPublic(rnd_vals);

        FPC<uint64_t> c(n);
        FPC<uint8_t> dc(n);

        Profiler profiler;
        profiler.start();

        ReLU(a, c, dc);

        profiler.accumulate("relu");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("4PC - relu (N=%d) - %f sec.\n", n, profiler.get_elapsed("relu") / 1000.0);
    }
}

TEST(EvalTest, Delphi_Convolutions) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<std::tuple<int, int, int, int, int> > dims = {
        std::make_tuple(32, 16, 16, 3, 32),
        std::make_tuple(32, 16, 16, 3, 32),
        std::make_tuple(16, 32, 32, 3, 16),
        std::make_tuple(8, 64, 64, 3, 8),
    };

    for (int i = 0; i < dims.size(); i++) {

        auto dim = dims[i];

        int im_size = std::get<0>(dim);
        int din = std::get<1>(dim);
        int dout = std::get<2>(dim);
        int f_size = std::get<3>(dim);
        int out_size = std::get<4>(dim);

        int N = 1;

        int a_size = N * din * im_size * im_size;
        random_vector(rnd_vals, a_size);
        TPC<uint64_t> a(a_size);
        a.setPublic(rnd_vals);

        int b_size = din * dout * f_size * f_size;
        random_vector(rnd_vals, b_size);
        TPC<uint64_t> b(b_size);
        b.setPublic(rnd_vals);

        TPC<uint64_t> c(N * dout * out_size * out_size);

        Profiler profiler;
        profiler.start();

	comm_profiler.clear();

        convolution(a, b, c, cutlass::conv::Operator::kFprop, N, im_size, im_size, f_size, din, dout, 1, 1, FLOAT_PRECISION);

        profiler.accumulate("conv");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - conv (N=1, Iw/h=%d, Din=%d, Dout=%d, f=%d) - %f sec.\n", im_size, din, dout, f_size, profiler.get_elapsed("conv") / 1000.0);
    	printf("TX comm (MB),%f\n", comm_profiler.get_comm_tx_bytes() / 1024.0 / 1024.0);
    	printf("RX comm (MB),%f\n", comm_profiler.get_comm_rx_bytes() / 1024.0 / 1024.0);
    }
}

TEST(EvalTest, GForce_Relu) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {100, 10000, 131072};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        TPC<uint64_t> a(n);
        a.setPublic(rnd_vals);

        TPC<uint64_t> c(n);
        TPC<uint8_t> dc(n);

        Profiler profiler;
        profiler.start();

	comm_profiler.clear();

        ReLU(a, c, dc);

        profiler.accumulate("relu");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - relu (N=%d) - %f sec.\n", n, profiler.get_elapsed("relu") / 1000.0);
    	printf("TX comm (MB),%f\n", comm_profiler.get_comm_tx_bytes() / 1024.0 / 1024.0);
    	printf("RX comm (MB),%f\n", comm_profiler.get_comm_rx_bytes() / 1024.0 / 1024.0);
    }
}

TEST(EvalTest, Sum_2PC_Profiling) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {32, 512, 8192, 16384};
    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        TPC<uint64_t> a(n);
        a.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> b(n);
        b.setPublic(rnd_vals);

        TPC<uint64_t> c(n);
        TPC<uint64_t> result(1);

        Profiler profiler;
        profiler.start();

        sum_and_reduce(a, b, c, result);

        profiler.accumulate("sum");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - sum (N=%d) - %f sec.\n", n, profiler.get_elapsed("sum") / 1000.0);
    }
}


TEST(EvalTest, Count_2PC_Profiling) {

    // if (partyNum >= 2) return;

    std::vector<double> rnd_vals;

    std::vector<int> N = {32, 512, 8192, 16384};
    uint64_t t = 5; // Threshold value set to 5 for testing

    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        TPC<uint64_t> a(n);
        a.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> b(n);
        b.setPublic(rnd_vals);

        TPC<uint64_t> result(1);  // To store the count result

        Profiler profiler;
        profiler.start();

        count_gt(a, b, result);

        profiler.accumulate("count");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - count (N=%d) - %f sec.\n", n, profiler.get_elapsed("count") / 1000.0);
    }
}

TEST(EvalTest, Billionaire_2PC_Profiling) {
    std::vector<double> rnd_vals;

    std::vector<int> N = {32, 512, 8192, 16384};

    for (int i = 0; i < N.size(); i++) {

        int n = N[i];

        random_vector(rnd_vals, n);
        TPC<uint64_t> a_cash(n);
        a_cash.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> a_property(n);
        a_property.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> a_stock(n);
        a_stock.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> b_cash(n);
        b_cash.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> b_property(n);
        b_property.setPublic(rnd_vals);

        random_vector(rnd_vals, n);
        TPC<uint64_t> b_stock(n);
        b_stock.setPublic(rnd_vals);

        TPC<uint64_t> result(1);  // To store the count result

        Profiler profiler;
        profiler.start();

    // call billionaire: counts how many indices where a_total > b_total
    billionaire(a_cash, a_property, a_stock, b_cash, b_property, b_stock, result);

    // Optionally reconstruct and print the count on the host for the test
    DeviceData<uint64_t> outVal(1);
    reconstruct(result, outVal);
    std::vector<uint64_t> host_out(1);
    thrust::copy(outVal.begin(), outVal.end(), host_out.begin());
    printf("billionaire count = %llu\n", (unsigned long long)host_out[0]);

        profiler.accumulate("count");

        // if (i == 0) continue; // sacrifice run to spin up GPU
        printf("2PC - billionaire (N=%d) - %f sec.\n", n, profiler.get_elapsed("count") / 1000.0);
    }
}