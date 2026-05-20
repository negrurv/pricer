#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatic std::vector <-> Python List conversion
#include "../include/models/heston.hpp"
#include "../include/system/thread_pool.hpp"

namespace py = pybind11;
using namespace pricer;

// A wrapper function to execute a batch run using our HPC Thread Pool
std::vector<double> batch_price_heston(const models::HestonParams& params, size_t num_contracts) {
    // static ensures the pool is created ONCE per Python process lifecycle
    static system::ThreadPool pool; 
    std::vector<double> results(num_contracts, 0.0);
    std::atomic<size_t> completed{0};
    
std::vector<system::OptionTask> tasks(num_contracts);
    for (size_t i = 0; i < num_contracts; ++i) {
        tasks[i].params = params;
        tasks[i].result_out = &results[i];
        tasks[i].completion_counter = &completed;
        
        // Inject a golden-ratio seed offset by the loop index
        tasks[i].path_seed = 0x9E3779B97F4A7C15ULL + i; 
        
        pool.submit(tasks[i]);
    }
    // Spin-wait for the batch to finish
    while (completed.load(std::memory_order_acquire) < num_contracts) {
        std::this_thread::yield();
    }

    return results;
}

PYBIND11_MODULE(pricer_core, m) {
    m.doc() = "Ultra-Low Latency C++ Options Pricer";

    py::class_<models::HestonParams>(m, "HestonParams")
        .def(py::init<>())
        .def_readwrite("S0", &models::HestonParams::S0)
        .def_readwrite("v0", &models::HestonParams::v0)
        .def_readwrite("r", &models::HestonParams::r)
        .def_readwrite("kappa", &models::HestonParams::kappa)
        .def_readwrite("theta", &models::HestonParams::theta)
        .def_readwrite("sigma", &models::HestonParams::sigma)
        .def_readwrite("rho", &models::HestonParams::rho)
        .def_readwrite("T", &models::HestonParams::T)
        .def_readwrite("K", &models::HestonParams::K)
        .def_readwrite("steps", &models::HestonParams::steps);

    m.def("batch_price_heston", &batch_price_heston, "Prices a batch of Heston options using the lock-free thread pool",
          py::arg("params"), py::arg("num_contracts"));
}
