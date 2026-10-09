use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir
        .parent()
        .expect("runtime parent")
        .join("kernels");
    let include_dir = kernels_dir.join("include");
    let src_dir = kernels_dir.join("src");

    let sources = [
        "matmul.cpp",
        "matvec.cpp",
        "linear.cpp",
        "linear_quant.cpp",
        "rmsnorm.cpp",
        "softmax.cpp",
        "rope.cpp",
        "activation.cpp",
    ];

    println!(
        "cargo:rerun-if-changed={}",
        include_dir.join("forge_kernels.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("simd.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("thread_pool.hpp").display()
    );
    for s in &sources {
        println!("cargo:rerun-if-changed={}", src_dir.join(s).display());
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(&include_dir)
        .include(&src_dir)
        .define("FORGE_KERNELS_STATIC", None)
        .warnings(false);

    for s in &sources {
        build.file(src_dir.join(s));
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "x86_64" {
        if build.get_compiler().is_like_msvc() {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2").flag("-mfma");
        }
    }

    build.compile("forge_kernels");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else if target_os == "windows" {
        let linker = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
        if linker == "gnu" {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }

    let want_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();
    let want_stub_feature = env::var("CARGO_FEATURE_CUDA_STUB").is_ok();
    let want_stub_env = env::var("FORGE_CUDA_STUB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if want_cuda && want_stub_feature {
        panic!(
            "enable only one Cargo feature: `cuda` (real libforge_cuda) or `cuda-stub` (compile-check)"
        );
    }

    // cuda-stub feature or FORGE_CUDA_STUB=1 with cuda → host stub for local compile checks.
    // Real `--features cuda` alone → requires libforge_cuda.
    if want_stub_feature || (want_stub_env && !want_cuda) {
        link_cuda_stub(&kernels_dir);
        println!("cargo:rustc-cfg=forge_cuda_backend");
        println!("cargo:rustc-cfg=forge_cuda_stub");
        println!("cargo:warning=linking forge_cuda STUB (compile-check only; no GPU)");
    } else if want_cuda {
        if want_stub_env {
            println!(
                "cargo:warning=FORGE_CUDA_STUB ignored because --features cuda requests real backend"
            );
        }
        link_real_cuda(&kernels_dir, &target_os);
        println!("cargo:rustc-cfg=forge_cuda_backend");
        println!("cargo:rustc-cfg=forge_cuda_real");
    }
}

fn link_cuda_stub(kernels_dir: &Path) {
    let cuda_inc = kernels_dir.join("cuda").join("include");
    let stub = kernels_dir
        .join("cuda")
        .join("src")
        .join("forge_cuda_stub.c");
    println!("cargo:rerun-if-changed={}", stub.display());
    println!(
        "cargo:rerun-if-changed={}",
        cuda_inc.join("forge_cuda.h").display()
    );
    cc::Build::new()
        .file(&stub)
        .include(&cuda_inc)
        .warnings(false)
        .compile("forge_cuda_stub");
}

fn link_real_cuda(kernels_dir: &Path, target_os: &str) {
    println!("cargo:rerun-if-env-changed=FORGE_CUDA_LIB_DIR");
    let cuda_build = env::var("FORGE_CUDA_LIB_DIR").unwrap_or_else(|_| {
        kernels_dir
            .join("build-cuda")
            .join("cuda")
            .to_string_lossy()
            .into_owned()
    });

    let search_dirs = [
        PathBuf::from(&cuda_build),
        PathBuf::from(&cuda_build).join("Release"),
        kernels_dir.join("build-cuda").join("cuda"),
        kernels_dir.join("build-cuda").join("cuda").join("Release"),
        kernels_dir.join("build-cuda"),
    ];

    let mut found_dir: Option<PathBuf> = None;
    for dir in &search_dirs {
        let a = dir.join("libforge_cuda.a");
        let b = dir.join("forge_cuda.lib");
        let c = dir.join("libforge_cuda.lib");
        if a.is_file() || b.is_file() || c.is_file() {
            found_dir = Some(dir.clone());
            break;
        }
    }

    let Some(lib_dir) = found_dir else {
        panic!(
            "\n\n\
             --features cuda requires a prebuilt libforge_cuda (real CUDA backend).\n\
             Searched under FORGE_CUDA_LIB_DIR={cuda_build:?} and kernels/build-cuda/**.\n\
             \n\
             Build on Linux with CUDA toolkit:\n\
               cmake -S kernels -B kernels/build-cuda -DFORGE_BUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release\n\
               cmake --build kernels/build-cuda -j\n\
               export FORGE_CUDA_LIB_DIR=$PWD/kernels/build-cuda/cuda\n\
               cargo build --release --features cuda\n\
             \n\
             For compile-only checks WITHOUT nvcc/GPU (stub FFI):\n\
               cargo build --features cuda-stub\n\
             \n\
             See docs/CUDA_TOOLCHAIN.md and scripts/verify_cuda_compile.sh\n"
        );
    };

    // Re-link after a kernel rebuild even when the Rust sources are unchanged.
    for name in ["libforge_cuda.a", "forge_cuda.lib", "libforge_cuda.lib"] {
        let archive = lib_dir.join(name);
        if archive.is_file() {
            println!("cargo:rerun-if-changed={}", archive.display());
        }
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=forge_cuda");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        // Prefer the system BFD/gold linker over rust-lld for CUDA static archives.
        println!("cargo:rustc-link-arg=-fuse-ld=bfd");
        // Common CUDA toolkit lib paths when not using full rpath setup.
        if let Ok(cuda_home) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
            let lib64 = PathBuf::from(&cuda_home).join("lib64");
            if lib64.is_dir() {
                println!("cargo:rustc-link-search=native={}", lib64.display());
            }
            let lib = PathBuf::from(&cuda_home).join("lib");
            if lib.is_dir() {
                println!("cargo:rustc-link-search=native={}", lib.display());
            }
        }
        for p in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib"] {
            if Path::new(p).is_dir() {
                println!("cargo:rustc-link-search=native={p}");
            }
        }
    }
}
