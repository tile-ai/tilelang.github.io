Search.setIndex({"docnames": ["deeplearning_operators/convolution", "deeplearning_operators/elementwise", "deeplearning_operators/flash_attention", "deeplearning_operators/flash_linear_attention", "deeplearning_operators/gemv", "deeplearning_operators/matmul", "deeplearning_operators/matmul_dequant", "deeplearning_operators/tmac_gpu", "get_started/Installation", "get_started/overview", "index", "language_ref/ast", "language_ref/primitives", "language_ref/tilelibrary", "privacy", "tutorials/annotate_memory_layout", "tutorials/auto_tuning", "tutorials/debug_tools_for_tilelang", "tutorials/jit_compilation", "tutorials/pipelining_computations_and_data_movements", "tutorials/writing_kernels_with_tilelibrary", "tutorials/writint_kernels_with_thread_primitives"], "filenames": ["deeplearning_operators/convolution.rst", "deeplearning_operators/elementwise.rst", "deeplearning_operators/flash_attention.rst", "deeplearning_operators/flash_linear_attention.rst", "deeplearning_operators/gemv.rst", "deeplearning_operators/matmul.rst", "deeplearning_operators/matmul_dequant.rst", "deeplearning_operators/tmac_gpu.rst", "get_started/Installation.rst", "get_started/overview.rst", "index.rst", "language_ref/ast.rst", "language_ref/primitives.rst", "language_ref/tilelibrary.rst", "privacy.rst", "tutorials/annotate_memory_layout.rst", "tutorials/auto_tuning.rst", "tutorials/debug_tools_for_tilelang.rst", "tutorials/jit_compilation.rst", "tutorials/pipelining_computations_and_data_movements.rst", "tutorials/writing_kernels_with_tilelibrary.rst", "tutorials/writint_kernels_with_thread_primitives.rst"], "titles": ["Convolution", "ElementWise Operators", "Flash Attention", "Flash Linear Attention", "General Matrix-Vector Multiplication (GEMV)", "General Matrix-Matrix Multiplication", "General Matrix-Matrix Multiplication with Dequantization", "TMAC: Look Up Table Based Mixed Precision Computing", "Installation Guide", "The Tile Language: A Brief Introduction", "\ud83d\udc4b Welcome to Tile Language", "Tile Language AST", "Tile Language: Primitives", "Tile Language: TileLibrary", "Privacy", "Annotate Memory Layout", "Auto-Tuning Techniques for Performance Optimization", "Debugging Tile Language Programs", "Just In Time Compilation", "Pipelining Computation and Data Movement", "Writing High-Performance Kernels with the Tile Library", "Annotating Memory Layout for Optimization"], "terms": {"prerequisit": 8, "via": [8, 9], "wheel": 8, "pypi": 8, "oper": [8, 9, 13, 17], "system": [8, 9], "ubuntu": 8, "20": 8, "04": 8, "later": [8, 13], "python": [8, 9, 10, 17], "version": [8, 17], "8": [8, 17], "cuda": [8, 9, 17], "11": 8, "0": [8, 13, 17], "The": [8, 10, 13, 17], "easiest": 8, "wai": [8, 13], "tilelang": [8, 9, 17], "i": [8, 9, 10, 13, 14, 17], "directli": [8, 9, 17], "To": 8, "latest": 8, "run": [8, 9, 17], "follow": [8, 9, 17], "command": 8, "termin": 8, "altern": 8, "you": [8, 13, 17], "mai": [8, 9, 17], "choos": [8, 9], "prebuilt": 8, "packag": 8, "avail": 8, "releas": 8, "page": 8, "dev0": 8, "4": [8, 17], "cu120": 8, "py3": 8, "none": 8, "ani": 8, "whl": 8, "github": [8, 10], "repositori": 8, "can": [8, 9, 13, 17], "git": 8, "http": 8, "com": 8, "tile": 8, "ai": 8, "after": [8, 9, 13, 17], "verifi": [8, 17], "c": [8, 9, 13, 17], "import": [8, 17], "print": 8, "__version__": 8, "linux": 8, "7": 8, "10": 8, "we": [8, 9, 13, 17], "recommend": 8, "docker": 8, "contain": [8, 9], "necessari": [8, 9, 10], "depend": [8, 9], "requir": [8, 17], "gpu": [8, 9, 10, 17], "all": [8, 13, 14, 17], "rm": 8, "ipc": 8, "host": 8, "nvcr": 8, "io": 8, "nvidia": [8, 9, 17], "pytorch": [8, 17], "23": 8, "01": 8, "step": 8, "thi": [8, 9, 13, 17], "process": [8, 9], "certain": 8, "pre": 8, "requisit": 8, "apach": 8, "which": [8, 9, 17], "debian": 8, "base": [8, 10, 17], "sudo": 8, "apt": 8, "get": [8, 17], "updat": [8, 13], "y": [8, 13], "python3": 8, "dev": 8, "setuptool": 8, "gcc": 8, "libtinfo": 8, "zlib1g": 8, "essenti": [8, 17], "cmake": 8, "libedit": 8, "libxml2": 8, "clone": 8, "recurs": 8, "cd": 8, "pleas": 8, "patient": 8, "take": [8, 17], "some": [8, 13, 17], "time": [8, 10], "If": [8, 13, 17], "want": 8, "develop": [8, 9, 10], "mode": 8, "e": [8, 9, 10, 13, 17], "current": [8, 13], "three": [8, 9, 17], "alreadi": 8, "have": [8, 9], "compat": 8, "note": [8, 9, 13, 17], "flag": 8, "includ": [8, 9, 17], "configur": 8, "option": 8, "creat": 8, "directori": 8, "specifi": 8, "exist": 8, "path": 8, "mkdir": 8, "dtvm_prebuild_path": 8, "g": [8, 9, 10, 13, 17], "workspac": 8, "make": [8, 9], "j": [8, 17], "16": [8, 17], "set": 8, "environ": 8, "variabl": [8, 13, 17], "pythonpath": 8, "lang": [8, 10], "modul": 8, "export": 8, "tvm_import_python_path": 8, "3rd": 8, "parti": 8, "framework": 8, "prefer": 8, "built": [8, 17], "instruct": 8, "ensur": 8, "fetch": 8, "copi": [8, 9, 10, 17], "file": [8, 9, 17], "enabl": [8, 9], "desir": [8, 9, 17], "backend": [8, 9], "llvm": [8, 9, 17], "cp": 8, "3rdparti": 8, "config": 8, "echo": 8, "use_llvm": 8, "ON": 8, "use_cuda": 8, "use_rocm": 8, "rocm": 8, "runtim": [8, 9], "output": [8, 17], "libtilelang": 8, "so": 8, "libtvm": 8, "libtvm_runtim": 8, "gener": [8, 9, 10], "For": [8, 9, 17], "simplifi": [8, 17], "bash": 8, "install_cuda": 8, "sh": 8, "install_amd": 8, "figur": 9, "below": [9, 17], "depict": 9, "how": [9, 17], "ar": [9, 17], "progress": [9, 17], "lower": [9, 17], "from": [9, 10, 13, 17], "high": [9, 10], "level": [9, 10, 17], "descript": 9, "specif": [9, 10, 17], "execut": [9, 17], "provid": [9, 17], "differ": 9, "target": [9, 17], "beginn": 9, "expert": 9, "user": [9, 14, 17], "each": [9, 13], "resid": 9, "pipelin": [9, 10, 17], "also": 9, "allow": [9, 10, 17], "mix": [9, 10], "within": [9, 13, 17], "same": 9, "kernel": [9, 10, 17], "work": 9, "whichev": 9, "abstract": 9, "best": 9, "suit": 9, "need": [9, 13, 17], "overview": 9, "unawar": 9, "intend": [9, 17], "who": 9, "write": [9, 10, 13, 17], "code": [9, 17], "independ": 9, "detail": 9, "goal": 9, "let": 9, "focu": [9, 10], "basic": 9, "logic": [9, 17], "without": [9, 10], "worri": 9, "about": 9, "hierarchi": 9, "optim": [9, 10, 13, 17], "yet": 9, "fulli": 9, "implement": [9, 13, 17], "awar": 9, "librari": [9, 10, 17], "design": [9, 10], "understand": 9, "perform": [9, 10, 13, 17], "consider": 9, "predefin": 9, "pattern": [9, 17], "variou": 9, "architectur": 9, "leverag": [9, 17], "readi": 9, "made": 9, "primit": [9, 10, 17], "dive": [9, 17], "low": [9, 10], "thread": [9, 13, 17], "highli": 9, "experienc": 9, "an": [9, 10, 17], "depth": 9, "characterist": 9, "coalesc": 9, "offer": 9, "direct": 9, "access": 9, "other": [9, 13, 17], "construct": 9, "fine": 9, "grain": 9, "control": 9, "critic": 9, "grant": 9, "maximum": 9, "flexibl": 9, "special": [9, 13, 17], "tailor": 9, "multi": 9, "core": 9, "comput": [9, 10, 17], "": [9, 13], "expertis": 9, "thei": [9, 13], "pure": 9, "incorpor": 9, "when": [9, 13, 17], "origin": 9, "expand": [9, 17], "call": [9, 17], "These": [9, 17], "encapsul": 9, "effici": 9, "explicitli": 9, "us": [9, 13, 17], "hand": 9, "data": [9, 10, 13, 14, 17], "layout": [9, 10], "synchron": [9, 17], "usag": 9, "irmodul": 9, "compos": 9, "intermedi": [9, 17], "represent": [9, 17], "ir": [9, 17], "captur": 9, "sourc": [9, 10], "hip": 9, "tune": [9, 10, 17], "amd": 9, "final": [9, 17], "correspond": 9, "devic": [9, 14, 17], "support": [9, 13], "multipl": [9, 10, 13], "extend": 9, "addit": 9, "gemm": [9, 10, 17], "concis": [9, 10], "matrix": [9, 10], "exampl": [9, 13], "illustr": [9, 17], "emploi": [9, 10], "placement": 9, "manag": 9, "movement": [9, 10], "In": [9, 10, 17], "particular": 9, "snippet": [9, 17], "demonstr": [9, 17], "global": [9, 13, 17], "share": [9, 13, 17], "regist": [9, 13, 17], "bandwidth": [9, 17], "util": 9, "reduc": [9, 13, 17], "latenc": 9, "overal": 9, "b": [9, 13, 17], "showcas": 9, "like": [9, 17], "syntax": [9, 10], "reason": 9, "friendli": 9, "At": 9, "heart": 9, "our": 9, "approach": [9, 17], "notion": 9, "first": [9, 17], "class": 9, "object": 9, "repres": [9, 13], "shape": [9, 13], "portion": 9, "own": 9, "manipul": 9, "warp": [9, 17], "block": 9, "equival": 9, "parallel": [9, 10, 17], "unit": 9, "matmul": [9, 17], "buffer": [9, 13, 17], "read": 9, "chunk": 9, "determin": 9, "block_m": [9, 17], "block_n": [9, 17], "block_k": [9, 17], "insid": [9, 13, 17], "loop": [9, 13], "With": 9, "t": [9, 10], "defin": [9, 17], "context": [9, 17], "index": [9, 17], "bx": [9, 17], "number": [9, 13], "help": [9, 17], "easier": 9, "automat": 9, "infer": [9, 13], "addition": 9, "manual": 9, "behavior": [9, 17], "hallmark": 9, "abil": 9, "place": 9, "rather": 9, "than": [9, 17], "leav": 9, "opaqu": 9, "pass": [9, 17], "expos": 9, "face": 9, "intrins": 9, "map": [9, 13], "physic": 9, "space": 9, "acceler": 9, "alloc_shar": [9, 10, 17], "fast": 9, "chip": 9, "storag": [9, 13], "ideal": 9, "cach": [9, 13], "dure": [9, 17], "significantli": [9, 17], "faster": 9, "between": [9, 13], "matric": 9, "load": [9, 13], "demand": 9, "improv": 9, "alloc_frag": [9, 10, 17], "accumul": 9, "fragment": [9, 13, 17], "By": [9, 10, 17], "keep": 9, "input": 9, "partial": 9, "sum": 9, "further": [9, 17], "minim": 9, "local": 9, "might": [9, 17], "seem": 9, "counterintuit": 9, "more": [9, 17], "abund": 9, "wherea": 9, "limit": [9, 17], "becaus": 9, "here": 9, "refer": [9, 17], "entir": [9, 17], "deriv": 9, "discuss": 9, "subsequ": 9, "section": 9, "transfer": 9, "furthermor": 9, "initi": [9, 17], "clear": [9, 10, 17], "fill": [9, 10], "assign": 9, "domain": 10, "streamlin": 10, "cpu": [10, 17], "dequant": 10, "flashattent": 10, "linearattent": 10, "underli": [10, 17], "compil": [10, 13, 17], "infrastructur": 10, "top": [10, 13], "tvm": [10, 17], "product": 10, "sacrif": 10, "state": 10, "art": 10, "instal": 10, "guid": 10, "pip": 10, "build": [10, 17], "A": [10, 13, 17], "brief": 10, "introduct": 10, "program": 10, "interfac": 10, "flow": 10, "model": 10, "annot": 10, "memori": [10, 13, 17], "debug": 10, "auto": [10, 17], "techniqu": 10, "just": 10, "elementwis": 10, "vector": [10, 13, 17], "gemv": 10, "flash": 10, "attent": 10, "linear": 10, "convolut": 10, "tmac": 10, "look": 10, "up": 10, "tabl": 10, "precis": 10, "ast": 10, "tilelibrari": 10, "reduce_max": 10, "reduce_sum": 10, "use_swizzl": 10, "arg": 13, "grid": 13, "size": 13, "3": [13, 17], "dimens": [13, 17], "num_thread": 13, "return": [13, 17], "blockidx": [13, 17], "launch": 13, "must": 13, "statement": [13, 17], "There": 13, "sequenti": 13, "prim": 13, "function": [13, 17], "dtype": [13, 17], "alloc": 13, "It": 13, "scope": 13, "should": 13, "dynam": 13, "whole": 13, "element": [13, 17], "distribut": 13, "store": 13, "partit": 13, "src": [13, 17], "dst": 13, "one": [13, 17], "bufferload": 13, "bufferregion": 13, "singl": 13, "start": 13, "point": 13, "param": 13, "sinc": 13, "know": 13, "region": 13, "zero": 13, "pad": 13, "detect": 13, "out": [13, 17], "boundari": 13, "transpose_a": 13, "transpose_b": 13, "polici": 13, "either": 13, "ha": 13, "constraint": 13, "length": 13, "reduct": 13, "axi": 13, "32": [13, 17], "fp16": 13, "multiplicand": 13, "case": [13, 17], "dim": [13, 17], "onli": [13, 17], "consid": [13, 17], "arbitrari": 13, "stop": 13, "num_stag": [13, 17], "convert": 13, "async": 13, "reorder": 13, "consum": 13, "produc": [13, 17], "nbsp": 13, "doubl": 13, "2": [13, 17], "noth": 13, "l2": 13, "x": 13, "serpentin": 13, "add": 13, "stai": 14, "collect": 14, "app": 14, "author": 17, "lei": 17, "wang": 17, "hereaft": 17, "transform": 17, "hardwar": 17, "through": 17, "sever": 17, "stage": 17, "undergo": 17, "see": 17, "engin": 17, "py": 17, "etc": 17, "respect": 17, "nvcc": 17, "encount": 17, "roughli": 17, "categori": 17, "fail": 17, "valid": 17, "error": 17, "result": 17, "incorrect": 17, "expect": 17, "theoret": 17, "tutori": 17, "focus": 17, "two": 17, "problem": 17, "often": 17, "vendor": 17, "profil": 17, "tool": 17, "nsight": 17, "rocprof": 17, "analysi": 17, "address": 17, "futur": 17, "materi": 17, "complet": 17, "def": 17, "m": 17, "n": 17, "k": 17, "float16": 17, "accum_dtyp": 17, "float": 17, "prim_func": 17, "main": 17, "ceildiv": 17, "128": 17, "a_shar": 17, "b_share": 17, "c_local": 17, "ko": 17, "back": 17, "1": 17, "func": 17, "1024": 17, "torch": 17, "jit_kernel": 17, "jitkernel": 17, "out_idx": 17, "test": 17, "randn": 17, "lowertileop": 17, "again": 17, "eventu": 17, "translat": 17, "instanc": 17, "occur": 17, "do": 17, "necessarili": 17, "jump": 17, "instead": 17, "inspect": 17, "them": 17, "where": 17, "simpl": 17, "1d": 17, "caus": 17, "commun": 17, "35": 17, "q": 17, "shape_q": 17, "seqlen_q": 17, "head": 17, "batch": 17, "num_split": 17, "bz": 17, "q_share": 17, "bid": 17, "hid": 17, "yield": 17, "root": 17, "codegen_cuda": 17, "cc": 17, "line": 17, "1257": 17, "valueerror": 17, "check": 17, "lane": 17, "v": 17, "ramp": 17, "indic": 17, "somewher": 17, "unsupport": 17, "wa": 17, "introduc": 17, "befor": 17, "right": 17, "device_mod": 17, "tir": 17, "filter": 17, "is_device_cal": 17, "mod": 17, "lowerdevicestorageaccessinfo": 17, "lowerintrin": 17, "kind": 17, "name": 17, "_ffi": 17, "get_global_func": 17, "tilelang_cuda": 17, "examin": 17, "calcul": 17, "incorrectli": 17, "reveal": 17, "handl": 17, "improperli": 17, "fix": 17, "refin": 17, "sometim": 17, "strategi": 17, "modifi": 17, "valu": 17, "codegen": 17, "rt_mod_cuda": 17, "std": 17, "string": 17, "cg": 17, "finish": 17, "const": 17, "f": 17, "registri": 17, "tilelang_callback_cuda_postproc": 17, "henc": 17, "intercept": 17, "register_func": 17, "_": 17, "insert": 17, "comment": 17, "instanti": 17, "simpli": 17, "Be": 17, "mind": 17, "concurr": 17, "show": 17, "found": 17, "codebas": 17, "test_tilelang_debug_print": 17, "debug_print_buff": 17, "shared_buf": 17, "get_profil": 17, "run_onc": 17, "stdout": 17, "mani": 17, "interleav": 17, "condit": 17, "nois": 17, "debug_print_buffer_condit": 17, "scalar": 17, "debug_print_value_condit": 17, "retriev": 17, "id": 17, "tid": 17, "get_thread_bind": 17, "content": 17, "still": 17, "debug_print_register_fil": 17, "register_buf": 17, "iter": 17, "ad": 17, "messag": 17, "prefix": 17, "suppli": 17, "distinguish": 17, "debug_print_msg": 17, "msg": 17, "hello": 17, "world": 17, "someth": 17, "threadidx": 17, "carefulli": 17, "quickli": 17, "diagnos": 17, "deviat": 17, "prong": 17, "suffici": 17, "resolv": 17, "advanc": 17, "analyz": 17, "occup": 17, "those": 17, "aspect": 17, "cover": 17, "document": 17}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"convolut": 0, "elementwis": 1, "oper": [1, 10], "flash": [2, 3], "attent": [2, 3], "linear": 3, "gener": [4, 5, 6, 17], "matrix": [4, 5, 6, 17], "vector": 4, "multipl": [4, 5, 6, 17], "gemv": 4, "dequant": 6, "tmac": 7, "look": 7, "up": 7, "tabl": 7, "base": [7, 9], "mix": 7, "precis": 7, "comput": [7, 19], "instal": 8, "guid": 8, "pip": 8, "build": 8, "from": 8, "sourc": [8, 17], "method": 8, "1": 8, "us": 8, "your": 8, "own": 8, "tvm": 8, "2": 8, "bundl": 8, "submodul": 8, "3": 8, "provid": 8, "script": 8, "The": 9, "tile": [9, 10, 11, 12, 13, 17, 20], "languag": [9, 10, 11, 12, 13, 17], "A": 9, "brief": 9, "introduct": 9, "program": [9, 17], "interfac": 9, "compil": [9, 18], "flow": 9, "model": 9, "declar": 9, "explicit": 9, "hardwar": 9, "memori": [9, 15, 21], "alloc": 9, "welcom": 10, "get": 10, "start": 10, "tutori": 10, "deep": 10, "learn": 10, "refer": 10, "privaci": [10, 14], "ast": 11, "primit": 12, "tilelibrari": 13, "t": [13, 17], "kernel": [13, 20], "alloc_shar": 13, "alloc_frag": 13, "copi": 13, "gemm": 13, "reduce_max": 13, "reduce_sum": 13, "parallel": 13, "pipelin": [13, 19], "clear": 13, "fill": 13, "use_swizzl": 13, "annot": [15, 21], "layout": [15, 21], "auto": 16, "tune": 16, "techniqu": 16, "perform": [16, 20], "optim": [16, 21], "debug": 17, "overview": 17, "exampl": 17, "issu": 17, "correct": 17, "post": 17, "process": 17, "callback": 17, "runtim": 17, "print": 17, "conclus": 17, "just": 18, "In": 18, "time": 18, "data": 19, "movement": 19, "write": 20, "high": 20, "librari": 20}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinx": 57}, "alltitles": {"Convolution": [[0, "convolution"]], "ElementWise Operators": [[1, "elementwise-operators"]], "Flash Attention": [[2, "flash-attention"]], "Flash Linear Attention": [[3, "flash-linear-attention"]], "General Matrix-Vector Multiplication (GEMV)": [[4, "general-matrix-vector-multiplication-gemv"]], "General Matrix-Matrix Multiplication": [[5, "general-matrix-matrix-multiplication"]], "General Matrix-Matrix Multiplication with Dequantization": [[6, "general-matrix-matrix-multiplication-with-dequantization"]], "TMAC: Look Up Table Based Mixed Precision Computing": [[7, "tmac-look-up-table-based-mixed-precision-computing"]], "Installation Guide": [[8, "installation-guide"]], "Installing with pip": [[8, "installing-with-pip"]], "Building from Source": [[8, "building-from-source"]], "Method 1: Install from Source (Using Your Own TVM Installation)": [[8, "method-1-install-from-source-using-your-own-tvm-installation"]], "Method 2: Install from Source (Using the Bundled TVM Submodule)": [[8, "method-2-install-from-source-using-the-bundled-tvm-submodule"]], "Method 3: Install Using the Provided Script": [[8, "method-3-install-using-the-provided-script"]], "The Tile Language: A Brief Introduction": [[9, "the-tile-language-a-brief-introduction"]], "Programming Interface": [[9, "programming-interface"]], "Programming Interfaces": [[9, "programming-interfaces"]], "Compilation Flow": [[9, "compilation-flow"]], "Tile-based Programming Model": [[9, "tile-based-programming-model"]], "Tile declarations": [[9, "tile-declarations"]], "Explicit Hardware Memory Allocation": [[9, "explicit-hardware-memory-allocation"]], "\ud83d\udc4b Welcome to Tile Language": [[10, "welcome-to-tile-language"]], "GET STARTED": [[10, null]], "TUTORIALS": [[10, null]], "DEEP LEARNING OPERATORS": [[10, null]], "LANGUAGE REFERENCE": [[10, null]], "Privacy": [[10, null], [14, "privacy"]], "Tile Language AST": [[11, "tile-language-ast"]], "Tile Language: Primitives": [[12, "tile-language-primitives"]], "Tile Language: TileLibrary": [[13, "tile-language-tilelibrary"]], "T.Kernel": [[13, "t-kernel"]], "T.alloc_shared": [[13, "t-alloc-shared"]], "T.alloc_fragment": [[13, "t-alloc-fragment"]], "T.copy": [[13, "t-copy"]], "T.gemm": [[13, "t-gemm"]], "T.reduce_max T.reduce_sum": [[13, "t-reduce-max-t-reduce-sum"]], "T.Parallel": [[13, "t-parallel"]], "T.Pipelined": [[13, "t-pipelined"]], "T.clear T.fill": [[13, "t-clear-t-fill"]], "T.use_swizzle": [[13, "t-use-swizzle"]], "Annotate Memory Layout": [[15, "annotate-memory-layout"]], "Auto-Tuning Techniques for Performance Optimization": [[16, "auto-tuning-techniques-for-performance-optimization"]], "Debugging Tile Language Programs": [[17, "debugging-tile-language-programs"]], "Overview": [[17, "overview"]], "Matrix Multiplication Example": [[17, "matrix-multiplication-example"]], "Debugging Generation Issues": [[17, "debugging-generation-issues"]], "Debugging Correctness Issues": [[17, "debugging-correctness-issues"]], "Post-Processing Callbacks for Generated Source": [[17, "post-processing-callbacks-for-generated-source"]], "Runtime Debug Prints with T.print": [[17, "runtime-debug-prints-with-t-print"]], "Conclusion": [[17, "conclusion"]], "Just In Time Compilation": [[18, "just-in-time-compilation"]], "Pipelining Computation and Data Movement": [[19, "pipelining-computation-and-data-movement"]], "Writing High-Performance Kernels with the Tile Library": [[20, "writing-high-performance-kernels-with-the-tile-library"]], "Annotating Memory Layout for Optimization": [[21, "annotating-memory-layout-for-optimization"]]}, "indexentries": {}})