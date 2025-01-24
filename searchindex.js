Search.setIndex({"docnames": ["get_started/Installation", "get_started/overview", "index", "language_ref/ast", "language_ref/primitives", "language_ref/tilelibrary", "privacy"], "filenames": ["get_started/Installation.rst", "get_started/overview.rst", "index.rst", "language_ref/ast.rst", "language_ref/primitives.rst", "language_ref/tilelibrary.rst", "privacy.rst"], "titles": ["Installation Guide", "The Tile Language: A Brief Introduction", "\ud83d\udc4b Welcome to Tile Language", "Tile Language AST", "Tile Language: Primitives", "Tile Language: TileLibrary", "Privacy"], "terms": {"prerequisit": 0, "via": [0, 1], "wheel": 0, "pypi": 0, "oper": [0, 1, 5], "system": [0, 1], "ubuntu": 0, "20": 0, "04": 0, "later": [0, 5], "python": [0, 1, 2], "version": 0, "8": 0, "cuda": [0, 1], "11": 0, "0": [0, 5], "The": [0, 2, 5], "easiest": 0, "wai": [0, 5], "tilelang": [0, 1], "i": [0, 1, 2, 5, 6], "directli": [0, 1], "To": 0, "latest": 0, "run": [0, 1], "follow": [0, 1], "command": 0, "termin": 0, "altern": 0, "you": [0, 5], "mai": [0, 1], "choos": [0, 1], "prebuilt": 0, "packag": 0, "avail": 0, "releas": 0, "page": 0, "dev0": 0, "4": 0, "cu120": 0, "py3": 0, "none": 0, "ani": 0, "whl": 0, "github": [0, 2], "repositori": 0, "can": [0, 1, 5], "git": 0, "http": 0, "com": 0, "tile": 0, "ai": 0, "after": [0, 1, 5], "verifi": 0, "c": [0, 1, 5], "import": 0, "print": 0, "__version__": 0, "linux": 0, "7": 0, "10": 0, "we": [0, 1, 5], "recommend": 0, "docker": 0, "contain": [0, 1], "necessari": [0, 1, 2], "depend": [0, 1], "requir": 0, "gpu": [0, 1, 2], "all": [0, 5, 6], "rm": 0, "ipc": 0, "host": 0, "nvcr": 0, "io": 0, "nvidia": [0, 1], "pytorch": 0, "23": 0, "01": 0, "step": 0, "thi": [0, 1, 5], "process": [0, 1], "certain": 0, "pre": 0, "requisit": 0, "apach": 0, "which": [0, 1], "debian": 0, "base": [0, 2], "sudo": 0, "apt": 0, "get": 0, "updat": [0, 5], "y": [0, 5], "python3": 0, "dev": 0, "setuptool": 0, "gcc": 0, "libtinfo": 0, "zlib1g": 0, "essenti": 0, "cmake": 0, "libedit": 0, "libxml2": 0, "clone": 0, "recurs": 0, "cd": 0, "pleas": 0, "patient": 0, "take": 0, "some": [0, 5], "time": 0, "If": [0, 5], "want": 0, "develop": [0, 1, 2], "mode": 0, "e": [0, 1, 2, 5], "current": [0, 5], "three": [0, 1], "alreadi": 0, "have": [0, 1], "compat": 0, "note": [0, 1, 5], "flag": 0, "includ": [0, 1], "configur": 0, "option": 0, "creat": 0, "directori": 0, "specifi": 0, "exist": 0, "path": 0, "mkdir": 0, "dtvm_prebuild_path": 0, "g": [0, 1, 2, 5], "workspac": 0, "make": [0, 1], "j": 0, "16": 0, "set": 0, "environ": 0, "variabl": [0, 5], "pythonpath": 0, "lang": [0, 2], "modul": 0, "export": 0, "tvm_import_python_path": 0, "3rd": 0, "parti": 0, "framework": 0, "prefer": 0, "built": 0, "instruct": 0, "ensur": 0, "fetch": 0, "copi": [0, 1, 2], "file": [0, 1], "enabl": [0, 1], "desir": [0, 1], "backend": [0, 1], "llvm": [0, 1], "cp": 0, "3rdparti": 0, "config": 0, "echo": 0, "use_llvm": 0, "ON": 0, "use_cuda": 0, "use_rocm": 0, "rocm": 0, "runtim": [0, 1], "output": 0, "libtilelang": 0, "so": 0, "libtvm": 0, "libtvm_runtim": 0, "gener": [0, 1], "For": [0, 1], "simplifi": 0, "bash": 0, "install_cuda": 0, "sh": 0, "install_amd": 0, "figur": 1, "below": 1, "depict": 1, "how": 1, "ar": 1, "progress": 1, "lower": 1, "from": [1, 2, 5], "high": [1, 2], "level": [1, 2], "descript": 1, "specif": [1, 2], "execut": 1, "provid": 1, "differ": 1, "target": 1, "beginn": 1, "expert": 1, "user": [1, 6], "each": [1, 5], "resid": 1, "pipelin": [1, 2], "also": 1, "allow": [1, 2], "mix": 1, "within": [1, 5], "same": 1, "kernel": [1, 2], "work": 1, "whichev": 1, "abstract": 1, "best": 1, "suit": 1, "need": [1, 5], "overview": 1, "unawar": 1, "intend": 1, "who": 1, "write": [1, 5], "code": 1, "independ": 1, "detail": 1, "goal": 1, "let": 1, "focu": [1, 2], "basic": 1, "logic": 1, "without": [1, 2], "worri": 1, "about": 1, "hierarchi": 1, "optim": [1, 2, 5], "yet": 1, "fulli": 1, "implement": [1, 5], "awar": 1, "librari": 1, "design": [1, 2], "understand": 1, "perform": [1, 2, 5], "consider": 1, "predefin": 1, "pattern": 1, "variou": 1, "architectur": 1, "leverag": 1, "readi": 1, "made": 1, "primit": [1, 2], "dive": 1, "low": [1, 2], "thread": [1, 5], "highli": 1, "experienc": 1, "an": [1, 2], "depth": 1, "characterist": 1, "coalesc": 1, "offer": 1, "direct": 1, "access": 1, "other": [1, 5], "construct": 1, "fine": 1, "grain": 1, "control": 1, "critic": 1, "grant": 1, "maximum": 1, "flexibl": 1, "special": [1, 5], "tailor": 1, "multi": 1, "core": 1, "comput": 1, "": [1, 5], "expertis": 1, "thei": [1, 5], "pure": 1, "incorpor": 1, "when": [1, 5], "origin": 1, "expand": 1, "call": 1, "These": 1, "encapsul": 1, "effici": 1, "explicitli": 1, "us": [1, 5], "hand": 1, "data": [1, 5, 6], "layout": 1, "synchron": 1, "usag": 1, "irmodul": 1, "compos": 1, "intermedi": 1, "represent": 1, "ir": 1, "captur": 1, "sourc": [1, 2], "hip": 1, "tune": 1, "amd": 1, "final": 1, "correspond": 1, "devic": [1, 6], "support": [1, 5], "multipl": [1, 5], "extend": 1, "addit": 1, "gemm": [1, 2], "concis": [1, 2], "matrix": 1, "exampl": [1, 5], "illustr": 1, "emploi": [1, 2], "placement": 1, "manag": 1, "movement": 1, "In": 1, "particular": 1, "snippet": 1, "demonstr": 1, "global": [1, 5], "share": [1, 5], "regist": [1, 5], "bandwidth": 1, "util": 1, "reduc": [1, 5], "latenc": 1, "overal": 1, "b": [1, 5], "showcas": 1, "like": 1, "syntax": [1, 2], "reason": 1, "friendli": 1, "At": 1, "heart": 1, "our": 1, "approach": 1, "notion": 1, "first": 1, "class": 1, "object": 1, "repres": [1, 5], "shape": [1, 5], "portion": 1, "own": 1, "manipul": 1, "warp": 1, "block": 1, "equival": 1, "parallel": [1, 2], "unit": 1, "matmul": 1, "buffer": [1, 5], "read": 1, "chunk": 1, "determin": 1, "block_m": 1, "block_n": 1, "block_k": 1, "insid": [1, 5], "loop": [1, 5], "With": 1, "t": [1, 2], "defin": 1, "context": 1, "index": 1, "bx": 1, "number": [1, 5], "help": 1, "easier": 1, "automat": 1, "infer": [1, 5], "addition": 1, "manual": 1, "behavior": 1, "hallmark": 1, "abil": 1, "place": 1, "rather": 1, "than": 1, "leav": 1, "opaqu": 1, "pass": 1, "expos": 1, "face": 1, "intrins": 1, "map": [1, 5], "physic": 1, "space": 1, "acceler": 1, "alloc_shar": [1, 2], "fast": 1, "chip": 1, "storag": [1, 5], "ideal": 1, "cach": [1, 5], "dure": 1, "significantli": 1, "faster": 1, "between": [1, 5], "matric": 1, "load": [1, 5], "demand": 1, "improv": 1, "alloc_frag": [1, 2], "accumul": 1, "fragment": [1, 5], "By": [1, 2], "keep": 1, "input": 1, "partial": 1, "sum": 1, "further": 1, "minim": 1, "local": 1, "might": 1, "seem": 1, "counterintuit": 1, "more": 1, "abund": 1, "wherea": 1, "limit": 1, "becaus": 1, "here": 1, "refer": 1, "entir": 1, "deriv": 1, "discuss": 1, "subsequ": 1, "section": 1, "transfer": 1, "furthermor": 1, "initi": 1, "clear": [1, 2], "fill": [1, 2], "assign": 1, "domain": 2, "streamlin": 2, "cpu": 2, "dequant": 2, "flashattent": 2, "linearattent": 2, "underli": 2, "compil": [2, 5], "infrastructur": 2, "top": [2, 5], "tvm": 2, "product": 2, "sacrif": 2, "state": 2, "art": 2, "instal": 2, "guid": 2, "pip": 2, "build": 2, "A": [2, 5], "brief": 2, "introduct": 2, "program": 2, "interfac": 2, "flow": 2, "model": 2, "ast": 2, "tilelibrari": 2, "reduce_max": 2, "reduce_sum": 2, "use_swizzl": 2, "arg": 5, "grid": 5, "size": 5, "3": 5, "dimens": 5, "num_thread": 5, "return": 5, "blockidx": 5, "launch": 5, "must": 5, "statement": 5, "There": 5, "sequenti": 5, "prim": 5, "function": 5, "dtype": 5, "alloc": 5, "memori": 5, "It": 5, "scope": 5, "should": 5, "dynam": 5, "whole": 5, "element": 5, "distribut": 5, "store": 5, "partit": 5, "src": 5, "dst": 5, "one": 5, "bufferload": 5, "bufferregion": 5, "singl": 5, "start": 5, "point": 5, "param": 5, "sinc": 5, "know": 5, "region": 5, "zero": 5, "pad": 5, "detect": 5, "out": 5, "boundari": 5, "transpose_a": 5, "transpose_b": 5, "polici": 5, "either": 5, "ha": 5, "constraint": 5, "length": 5, "reduct": 5, "axi": 5, "32": 5, "fp16": 5, "multiplicand": 5, "case": 5, "dim": 5, "onli": 5, "consid": 5, "vector": 5, "arbitrari": 5, "stop": 5, "num_stag": 5, "convert": 5, "async": 5, "reorder": 5, "consum": 5, "produc": 5, "nbsp": 5, "doubl": 5, "2": 5, "noth": 5, "l2": 5, "x": 5, "serpentin": 5, "add": 5, "stai": 6, "collect": 6, "app": 6}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"instal": 0, "guid": 0, "pip": 0, "build": 0, "from": 0, "sourc": 0, "method": 0, "1": 0, "us": 0, "your": 0, "own": 0, "tvm": 0, "2": 0, "bundl": 0, "submodul": 0, "3": 0, "provid": 0, "script": 0, "The": 1, "tile": [1, 2, 3, 4, 5], "languag": [1, 2, 3, 4, 5], "A": 1, "brief": 1, "introduct": 1, "program": 1, "interfac": 1, "compil": 1, "flow": 1, "base": 1, "model": 1, "declar": 1, "explicit": 1, "hardwar": 1, "memori": 1, "alloc": 1, "welcom": 2, "get": 2, "start": 2, "refer": 2, "privaci": [2, 6], "ast": 3, "primit": 4, "tilelibrari": 5, "t": 5, "kernel": 5, "alloc_shar": 5, "alloc_frag": 5, "copi": 5, "gemm": 5, "reduce_max": 5, "reduce_sum": 5, "parallel": 5, "pipelin": 5, "clear": 5, "fill": 5, "use_swizzl": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"Installation Guide": [[0, "installation-guide"]], "Installing with pip": [[0, "installing-with-pip"]], "Building from Source": [[0, "building-from-source"]], "Method 1: Install from Source (Using Your Own TVM Installation)": [[0, "method-1-install-from-source-using-your-own-tvm-installation"]], "Method 2: Install from Source (Using the Bundled TVM Submodule)": [[0, "method-2-install-from-source-using-the-bundled-tvm-submodule"]], "Method 3: Install Using the Provided Script": [[0, "method-3-install-using-the-provided-script"]], "The Tile Language: A Brief Introduction": [[1, "the-tile-language-a-brief-introduction"]], "Programming Interface": [[1, "programming-interface"]], "Programming Interfaces": [[1, "programming-interfaces"]], "Compilation Flow": [[1, "compilation-flow"]], "Tile-based Programming Model": [[1, "tile-based-programming-model"]], "Tile declarations": [[1, "tile-declarations"]], "Explicit Hardware Memory Allocation": [[1, "explicit-hardware-memory-allocation"]], "\ud83d\udc4b Welcome to Tile Language": [[2, "welcome-to-tile-language"]], "GET STARTED": [[2, null]], "LANGUAGE REFERENCE": [[2, null]], "Privacy": [[2, null], [6, "privacy"]], "Tile Language AST": [[3, "tile-language-ast"]], "Tile Language: Primitives": [[4, "tile-language-primitives"]], "Tile Language: TileLibrary": [[5, "tile-language-tilelibrary"]], "T.Kernel": [[5, "t-kernel"]], "T.alloc_shared": [[5, "t-alloc-shared"]], "T.alloc_fragment": [[5, "t-alloc-fragment"]], "T.copy": [[5, "t-copy"]], "T.gemm": [[5, "t-gemm"]], "T.reduce_max T.reduce_sum": [[5, "t-reduce-max-t-reduce-sum"]], "T.Parallel": [[5, "t-parallel"]], "T.Pipelined": [[5, "t-pipelined"]], "T.clear T.fill": [[5, "t-clear-t-fill"]], "T.use_swizzle": [[5, "t-use-swizzle"]]}, "indexentries": {}})