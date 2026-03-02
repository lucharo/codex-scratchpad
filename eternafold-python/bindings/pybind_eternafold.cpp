#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <iostream>

// EternaFold headers
#include "Options.hpp"
#include "FileDescription.hpp"
#include "ComputationEngine.hpp"
#include "Config.hpp"
#include "Utilities.hpp"
#include "ComputationWrapper.hpp"
#include "InferenceEngine.hpp"
#include "ParameterManager.hpp"
#include "OptimizationWrapper.hpp"
#include "SStruct.hpp"
#include "Defaults.ipp"

namespace py = pybind11;

namespace {

void init_options(Options &options) {
    options.SetStringValue("training_mode", "");
    options.SetBoolValue("verbose_output", false);
    options.SetRealValue("log_base", 1.0);
    options.SetBoolValue("viterbi_parsing", false);
    options.SetBoolValue("allow_noncomplementary", false);
    options.SetStringValue("parameter_filename", "");
    options.SetBoolValue("use_constraints", false);
    options.SetBoolValue("centroid_estimator", false);
    options.SetBoolValue("use_evidence", false);
    options.SetStringValue("output_parens_destination", "");
    options.SetStringValue("output_bpseq_destination", "");
    options.SetRealValue("output_posteriors_cutoff", 0);
    options.SetStringValue("output_posteriors_destination", "");
    options.SetBoolValue("partition_function_only", false);
    options.SetBoolValue("gradient_sanity_check", false);
    options.SetRealValue("holdout_ratio", 0);
    options.SetStringValue("train_examplefile", "");
    options.SetStringValue("train_initweights_filename", "");
    options.SetStringValue("train_priorweights_filename", "");
    options.SetIntValue("num_data_sources", 0);
    options.SetIntValue("batch_size", 1);
    options.SetRealValue("s0", 0.0001);
    options.SetRealValue("s1", 0);
    options.SetRealValue("kappa", 1.0);
}

// Suppress cout during computation
struct ScopedSilence {
    std::streambuf* old_buf;
    std::ostringstream sink;
    ScopedSilence() : old_buf(std::cout.rdbuf(sink.rdbuf())) {}
    ~ScopedSilence() { std::cout.rdbuf(old_buf); }
};

float compute_pfunc(const std::string &sequence, const std::string &constraints,
                    const std::string &param_file) {
    ScopedSilence silence;

    Options options;
    init_options(options);
    options.SetBoolValue("use_constraints", true);
    options.SetBoolValue("partition_function_only", true);

    std::string constr = (constraints == "?")
        ? std::string(sequence.length(), '?')
        : constraints;

    ParameterManager<float> parameter_manager;
    InferenceEngine<float> inference_engine(
        options.GetBoolValue("allow_noncomplementary"), 0,
        options.GetRealValue("kappa"));
    inference_engine.RegisterParameters(parameter_manager);

    // Load parameters from file or use defaults
    std::vector<float> w;
    if (!param_file.empty()) {
        parameter_manager.ReadFromFile(param_file, w);
    } else {
        w = GetDefaultComplementaryValues<float>();
    }
    inference_engine.LoadValues(w);

    SStruct sstruct;
    sstruct.LoadAPI(sequence, constr);
    inference_engine.LoadSequence(sstruct);
    inference_engine.UseConstraints(sstruct.GetMapping());

    std::vector<FileDescription> descriptions;
    ComputationEngine<float> computation_engine(
        options, descriptions, inference_engine, parameter_manager);

    inference_engine.ComputeInside();
    float Z = inference_engine.ComputeLogPartitionCoefficient();
    inference_engine.ComputeOutside();
    inference_engine.ComputePosterior();

    return Z;
}

std::string compute_predict(const std::string &sequence,
                            const std::string &constraints,
                            const std::string &param_file,
                            float gamma) {
    ScopedSilence silence;

    Options options;
    init_options(options);
    options.SetBoolValue("use_constraints", true);
    options.SetBoolValue("partition_function_only", true);

    std::string constr = (constraints == "?")
        ? std::string(sequence.length(), '?')
        : constraints;

    ParameterManager<float> parameter_manager;
    InferenceEngine<float> inference_engine(
        options.GetBoolValue("allow_noncomplementary"), 0,
        options.GetRealValue("kappa"));
    inference_engine.RegisterParameters(parameter_manager);

    std::vector<float> w;
    if (!param_file.empty()) {
        parameter_manager.ReadFromFile(param_file, w);
    } else {
        w = GetDefaultComplementaryValues<float>();
    }
    inference_engine.LoadValues(w);

    SStruct sstruct;
    sstruct.LoadAPI(sequence, constr);
    inference_engine.LoadSequence(sstruct);
    inference_engine.UseConstraints(sstruct.GetMapping());

    std::vector<FileDescription> descriptions;
    ComputationEngine<float> computation_engine(
        options, descriptions, inference_engine, parameter_manager);

    SStruct solution(sstruct);

    inference_engine.ComputeInside();
    inference_engine.ComputeOutside();
    inference_engine.ComputePosterior();

    solution.SetMapping(inference_engine.PredictPairingsPosterior(gamma));

    return solution.ReturnMappingAsParens();
}

std::vector<std::vector<float>> compute_bpps(const std::string &sequence,
                                              const std::string &constraints,
                                              const std::string &param_file) {
    ScopedSilence silence;

    Options options;
    init_options(options);
    options.SetBoolValue("use_constraints", true);
    options.SetBoolValue("partition_function_only", true);

    std::string constr = (constraints == "?")
        ? std::string(sequence.length(), '?')
        : constraints;

    ParameterManager<float> parameter_manager;
    InferenceEngine<float> inference_engine(
        options.GetBoolValue("allow_noncomplementary"), 0,
        options.GetRealValue("kappa"));
    inference_engine.RegisterParameters(parameter_manager);

    std::vector<float> w;
    if (!param_file.empty()) {
        parameter_manager.ReadFromFile(param_file, w);
    } else {
        w = GetDefaultComplementaryValues<float>();
    }
    inference_engine.LoadValues(w);

    SStruct sstruct;
    sstruct.LoadAPI(sequence, constr);
    inference_engine.LoadSequence(sstruct);
    inference_engine.UseConstraints(sstruct.GetMapping());

    std::vector<FileDescription> descriptions;
    ComputationEngine<float> computation_engine(
        options, descriptions, inference_engine, parameter_manager);

    inference_engine.ComputeInside();
    inference_engine.ComputeOutside();
    inference_engine.ComputePosterior();

    // Get posterior probabilities from flat upper-triangular array
    // GetPosterior returns a new float[] of size (L+1)*(L+2)/2
    // indexed as upper triangular: for row i (0..L), cols j (i..L)
    // Positions are 1-based in the engine (0 is boundary)
    const int n = sequence.length();
    const int size = n + 1;  // L+1
    std::vector<std::vector<float>> bpp_matrix(n, std::vector<float>(n, 0.0f));

    float *posterior = inference_engine.GetPosterior(0.0f);

    // Walk the flat upper-triangular array
    int idx = 0;
    for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
            float p = posterior[idx++];
            // Convert 1-based positions to 0-based
            if (p > 0.0f && i >= 1 && j >= 1 && i <= n && j <= n) {
                bpp_matrix[i-1][j-1] = p;
                bpp_matrix[j-1][i-1] = p;
            }
        }
    }

    delete[] posterior;  // GetPosterior allocates with new[]
    return bpp_matrix;
}

} // anonymous namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "EternaFold RNA secondary structure prediction (C++ core)";

    m.def("pfunc", &compute_pfunc,
        py::arg("sequence"),
        py::arg("constraints") = "?",
        py::arg("param_file") = "",
        "Compute the log partition function for an RNA sequence.\n\n"
        "Args:\n"
        "    sequence: RNA sequence (ACGU)\n"
        "    constraints: Structure constraints ('?' for unconstrained)\n"
        "    param_file: Path to parameter file (empty for defaults)\n\n"
        "Returns:\n"
        "    Log partition coefficient (float)");

    m.def("predict", &compute_predict,
        py::arg("sequence"),
        py::arg("constraints") = "?",
        py::arg("param_file") = "",
        py::arg("gamma") = 6.0f,
        "Predict the MEA secondary structure for an RNA sequence.\n\n"
        "Args:\n"
        "    sequence: RNA sequence (ACGU)\n"
        "    constraints: Structure constraints ('?' for unconstrained)\n"
        "    param_file: Path to parameter file (empty for defaults)\n"
        "    gamma: Sensitivity/specificity tradeoff (default 6.0)\n\n"
        "Returns:\n"
        "    Dot-bracket structure string");

    m.def("bpps", &compute_bpps,
        py::arg("sequence"),
        py::arg("constraints") = "?",
        py::arg("param_file") = "",
        "Compute base pair probability matrix for an RNA sequence.\n\n"
        "Args:\n"
        "    sequence: RNA sequence (ACGU)\n"
        "    constraints: Structure constraints ('?' for unconstrained)\n"
        "    param_file: Path to parameter file (empty for defaults)\n\n"
        "Returns:\n"
        "    NxN list of lists with base pair probabilities");
}
