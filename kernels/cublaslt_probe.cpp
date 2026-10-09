// Experimental cuBLASLt algorithm search, not enabled by the submission.
// Written against NVIDIA's public cuBLAS 12.8 API; no sample source is copied.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <sstream>
#include <vector>

static void check(cublasStatus_t status) {
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(status));
}

struct Preference {
    cublasLtMatmulPreference_t value = nullptr;
    Preference() { check(cublasLtMatmulPreferenceCreate(&value)); }
    ~Preference() { if(value) cublasLtMatmulPreferenceDestroy(value); }
};

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    at::Tensor workspace;
    int device;
    Plan(const at::Tensor& x, const at::Tensor& w, int limit) : device(x.get_device()) {
      try {
        check(cublasLtCreate(&handle));
        check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t trans = CUBLAS_OP_T;
        check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
        const int64_t m=x.size(0), n=w.size(0), k=x.size(1);
        check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, m, k, x.stride(0)));
        check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, n, k, w.stride(0)));
        check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, m, n, n));
        cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        for (auto layout : {a, b, c})
            check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
        Preference preference;
        auto pref=preference.value;
        size_t bytes = 0;
        check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &bytes, sizeof(bytes)));
        // Views can have a padded leading dimension. Communicate actual pointer
        // alignment, not a fictitious guarantee copied from contiguous examples.
        auto alignment = [](const void* ptr) {
            uint32_t size=1; const auto address=reinterpret_cast<uintptr_t>(ptr);
            while (size < 256 && address % (size * 2) == 0) size *= 2;
            return size;
        };
        uint32_t aa=alignment(x.data_ptr()), ab=alignment(w.data_ptr()), ac=256;
        check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &aa, sizeof(aa)));
        check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &ab, sizeof(ab)));
        check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &ac, sizeof(ac)));
        check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &ac, sizeof(ac)));
        candidates.resize(limit);
        int count=0;
        auto status=cublasLtMatmulAlgoGetHeuristic(handle, op, a, b, c, c, pref,
                                                  limit, candidates.data(), &count);
        check(status);
        candidates.resize(count);
        workspace=at::empty({int64_t(bytes)}, x.options().dtype(at::kByte));
      } catch (...) {
        release();
        throw;
      }
    }
    ~Plan() {
        // Workspace and descriptors live as long as cached algorithms. No inputs,
        // weights or outputs are retained, and destruction occurs before unload.
        release();
    }
    void release() noexcept {
        if(a) { cublasLtMatrixLayoutDestroy(a); a=nullptr; }
        if(b) { cublasLtMatrixLayoutDestroy(b); b=nullptr; }
        if(c) { cublasLtMatrixLayoutDestroy(c); c=nullptr; }
        if(op) { cublasLtMatmulDescDestroy(op); op=nullptr; }
        if(handle) { cublasLtDestroy(handle); handle=nullptr; }
    }
};

// Handles are thread-local. Only zero-workspace algorithms are queried: there
// is no scratch buffer to race when graph capture uses a different CUDA stream.
static thread_local std::map<std::string, std::unique_ptr<Plan>> plans;
static std::string key(const at::Tensor& x, const at::Tensor& w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device()==w.device(), "same CUDA device required");
    TORCH_CHECK(x.scalar_type()==at::kHalf && w.scalar_type()==at::kHalf, "fp16 inputs required");
    TORCH_CHECK(x.dim()==2 && w.dim()==2 && x.size(1)==w.size(1), "matrix dimensions mismatch");
    TORCH_CHECK(x.stride(1)==1 && w.stride(1)==1, "inner dimension must be contiguous");
    std::ostringstream s;
    s << x.get_device() << ':' << x.size(0) << ':' << w.size(0) << ':' << x.size(1)
      << ':' << x.stride(0) << ':' << w.stride(0);
    // Avoid a plan with more optimistic alignment than a later offset view.
    s << ':' << (reinterpret_cast<uintptr_t>(x.data_ptr()) % 256)
      << ':' << (reinterpret_cast<uintptr_t>(w.data_ptr()) % 256);
    return s.str();
}

static std::vector<std::vector<int64_t>> algorithms(const at::Tensor& x, const at::Tensor& w, int limit) {
    c10::cuda::CUDAGuard guard(x.device());
    const auto k=key(x,w);
    if (!plans.count(k)) plans[k]=std::make_unique<Plan>(x,w,limit);
    std::vector<std::vector<int64_t>> result;
    for (const auto& h: plans.at(k)->candidates) {
        std::vector<int64_t> row={int64_t(h.state), int64_t(h.workspaceSize)};
        for (auto attr : {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
                          CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                          CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_STAGES_ID}) {
            int value=0; size_t written=0;
            auto status=cublasLtMatmulAlgoConfigGetAttribute(&h.algo, attr, &value, sizeof(value), &written);
            row.push_back(status==CUBLAS_STATUS_SUCCESS ? value : -1);
        }
        result.push_back(row);
    }
    return result;
}

static at::Tensor matmul(const at::Tensor& x, const at::Tensor& w, int index) {
    c10::cuda::CUDAGuard guard(x.device());
    const auto k=key(x,w);
    TORCH_CHECK(plans.count(k), "call algorithms before timing or graph capture");
    const auto& p=plans.at(k);
    TORCH_CHECK(index>=0 && index<int(p->candidates.size()), "invalid algorithm index");
    auto out=at::empty({x.size(0), w.size(0)}, x.options().dtype(at::kFloat));
    float alpha=1.0f, beta=0.0f;
    check(cublasLtMatmul(p->handle, p->op, &alpha, x.data_ptr(), p->a,
                        w.data_ptr(), p->b, &beta, out.data_ptr(), p->c,
                        out.data_ptr(), p->c, &p->candidates[index].algo,
                        p->workspace.data_ptr(), p->workspace.numel(),
                        at::cuda::getCurrentCUDAStream(x.get_device())));
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("algorithms", &algorithms);
    m.def("matmul", &matmul);
    m.def("clear", [] { plans.clear(); });
    m.def("tensor_keys", [](pybind11::list objects) {
        pybind11::list keys;
        for (const auto& object : objects) {
            const auto tensor=pybind11::cast<at::Tensor>(object);
            keys.append(pybind11::make_tuple(
                reinterpret_cast<uintptr_t>(object.ptr()),
                tensor.unsafeGetTensorImpl()->version_counter().current_version(),
                reinterpret_cast<uintptr_t>(tensor.data_ptr()),
                int(tensor.device().type()), int(tensor.device().index()),
                int(tensor.scalar_type())));
        }
        return keys;
    });
}
