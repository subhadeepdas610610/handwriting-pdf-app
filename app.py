import base64
import io
from io import BytesIO
import json
from pathlib import Path
import random
import re
import zipfile
import cv2
import numpy as np
import pymupdf as fitz
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

DEFAULT_GLYPHS_DIR = Path("glyphs")
PRESETS_FILE = Path("glyph_studio_presets.json")

PAGE_W, PAGE_H = 1240, 1754
MARGIN_TOP = 130
MARGIN_BOTTOM = 120

ASCENDERS = set("bdfhklt")
DESCENDERS = set("fgjpqyz")
SMALL_PUNCT = set(".,:;!-'\"`")

INK_PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (28, 30, 36),
    "blue": (28, 62, 148),
    "red": (168, 34, 42),
    "green": (26, 112, 64),
}

SHEET_1_LABELS: list[str] = (
    list("abcdefghijklmnopqrstuvwxyz")
    + [f"upper_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    + list("0123456789")
    + [".", ",", "question", "!", "-", "+", "=", ":", ";", "quote_single", "quote_double"]
)

SHEET_2_DOUBLE_PAIRS: list[str] = [
    "pp", "tt", "ll", "ee", "oo", "ss", "ff", "rr", "mm", "nn",
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ch", "sh", "wh", "ck", "qu", "ve", "de", "ce", "ge", "pe",
    "me", "ne", "we", "be", "ke", "fe", "ly", "ty", "ry", "sy",
    "my", "by", "do", "go", "no", "so", "up", "us", "am", "if",
]

SHEET_3_TRIPLE_WORDS: list[str] = [
    "the", "and", "ing", "ion", "ent", "for", "tha", "ter", "hat", "his",
    "ere", "her", "ate", "ver", "all", "con", "ted", "com", "not", "are",
    "but", "you", "can", "had", "was", "one", "our", "out", "day", "get",
    "has", "him", "how", "man", "new", "now", "old", "see", "two", "way",
    "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
    "any", "may", "own", "per", "pre", "pro", "sub", "dis", "mis", "non",
    "ove", "end", "act", "age", "air", "art", "ask", "bad", "big", "far",
]

SHEET_4_COMMON_4: list[str] = [
    "that", "with", "have", "this", "will", "your", "from", "they", "know", "want",
    "been", "good", "much", "some", "time", "very", "when", "come", "here", "just",
    "like", "long", "make", "many", "more", "only", "over", "such", "take", "than",
    "them", "well", "were", "what", "into", "year", "most", "also", "back", "even",
    "give", "work", "tion", "ment", "ness", "able", "ible", "less", "ship", "ward",
    "ight", "ough", "ould", "each", "page", "line", "note", "word", "text", "hand",
    "size", "part", "form", "user", "file", "data", "code", "list", "look", "keep",
]

SHEET_5_COMMON_5: list[str] = [
    "which", "their", "there", "about", "would", "these", "other", "words", "could", "write",
    "first", "water", "after", "where", "right", "think", "three", "years", "place", "sound",
    "great", "again", "still", "every", "small", "found", "those", "never", "under", "might",
    "while", "house", "world", "below", "asked", "going", "large", "until", "along", "shall",
    "being", "often", "earth", "began", "since", "study", "night", "light", "above", "paper",
    "parts", "young", "story", "point", "times", "heard", "whole", "white", "given", "means",
    "music", "miles", "thing", "today", "later", "using", "money", "lines", "order", "group",
]

SHEET_6_MATH_GREEK: list[str] = [
    "math_alpha", "math_beta", "math_gamma", "math_delta", "math_epsilon",
    "math_zeta", "math_eta", "math_theta", "math_iota", "math_kappa",
    "math_lambda", "math_mu", "math_nu", "math_xi", "math_pi",
    "math_rho", "math_sigma", "math_tau", "math_upsilon", "math_phi",
    "math_chi", "math_psi", "math_omega",
    "math_Gamma", "math_Delta", "math_Theta", "math_Lambda", "math_Xi",
    "math_Pi", "math_Sigma", "math_Phi", "math_Psi", "math_Omega",
]

SHEET_7_MATH_OPERATORS: list[str] = [
    "math_int", "math_iint", "math_oint", "math_sum", "math_prod",
    "math_sqrt", "math_partial", "math_nabla", "math_infty", "math_pm",
    "math_mp", "math_times", "math_div", "math_cdot", "math_approx",
    "math_neq", "math_equiv", "math_leq", "math_geq", "math_ll",
    "math_gg", "math_propto", "math_rarr", "math_larr", "math_harr",
    "math_Rarr", "math_Harr", "math_forall", "math_exists", "math_in",
    "math_notin", "math_subset", "math_cup", "math_cap", "math_emptyset",
    "math_therefore", "math_because",
]

SHEET_8_MATH_BRACKETS_NOTATION: list[str] = [
    "lparen", "rparen", "lbrack", "rbrack", "lbrace", "rbrace",
    "vbar", "dblbar", "langle", "rangle", "slash", "bslash",
    "caret", "underscore", "degree", "prime", "dprime", "angle",
    "perp", "parallel", "triangle", "square_op", "tilde_op", "hat_op",
    "bar_op", "vec_op", "dot_op", "ddot_op", "star_op", "circ_op",
]

SHEET_9_MATH_COMMON_TERMS: list[str] = [
    "lig_dx", "lig_dy", "lig_dt", "lig_dz", "lig_du", "lig_dv",
    "lig_sin", "lig_cos", "lig_tan", "lig_cot", "lig_sec", "lig_csc",
    "lig_log", "lig_ln", "lig_exp", "lig_lim", "lig_max", "lig_min",
    "lig_sup", "lig_inf", "lig_det", "lig_dim", "lig_ker", "lig_arg",
    "lig_mod", "lig_deg", "lig_gcd", "lig_lcm", "lig_Var", "lig_Cov",
]

SHEET_10_FLOW_NODES: list[str] = [
    "flow_oval", "flow_rect", "flow_diamond", "flow_parallelogram",
    "flow_circle", "flow_cylinder", "flow_hexagon", "flow_document",
    "flow_cloud", "flow_subproc",
]

SHEET_11_FLOW_ARROWS: list[str] = [
    "flow_arr_right", "flow_arr_left", "flow_arr_down", "flow_arr_up",
    "flow_arr_bi", "flow_elbow_rd", "flow_elbow_dr", "flow_arr_dashed",
    "flow_line_h", "flow_line_v",
]

# Sheets 12-18: Additional Common English Words, Clusters & Patterns
SHEET_12_ENG_6LETTER: list[str] = [
    "system", "return", "output", "should", "within", "number", "create", "values",
    "method", "object", "string", "memory", "process", "action", "result", "source",
    "target", "format", "status", "access", "buffer", "packet", "server", "client",
    "window", "border", "center", "letter", "figure", "reason", "change", "follow",
    "before", "during", "around", "across", "almost", "always", "enough", "either",
    "rather", "better", "single", "double", "simple", "direct", "normal", "custom",
    "public", "import", "export", "module", "script", "syntax", "parser", "tokens",
]

SHEET_13_ENG_7LETTER: list[str] = [
    "because", "between", "through", "without", "example", "default", "network", "request",
    "response", "handler", "element", "content", "section", "chapter", "summary", "problem",
    "solution", "concept", "feature", "version", "release", "testing", "loading", "parsing",
    "storage", "latency", "routing", "address", "session", "timeout", "integer", "boolean",
    "pointer", "closure", "promise", "runtime", "package", "library", "command", "terminal",
    "diagram", "picture", "spacing", "aligned", "preview", "sidebar", "toolbar", "dataset",
    "trained", "weights", "epsilon", "gradient", "theorem", "formula", "proofs", "derived",
]

SHEET_14_ENG_AFFIXES: list[str] = [
    "ing", "tion", "ment", "ness", "able", "ible", "less", "ship", "ward", "ized",
    "ised", "ally", "ical", "ious", "eous", "ty", "er", "or", "est", "ed",
    "un_", "re_", "de_", "mis_", "over_", "under_", "inter_", "trans_", "sub_", "super_",
    "non_", "anti_", "auto_", "semi_", "multi_", "micro_", "macro_", "proto_", "pseudo_", "co_",
]

SHEET_15_ENG_VERBS_ADJ: list[str] = [
    "check", "build", "write", "read", "run", "stop", "start", "open", "close", "save",
    "load", "send", "recv", "sort", "find", "scan", "walk", "push", "pop", "peek",
    "insert", "delete", "update", "select", "filter", "group", "order", "join", "merge", "split",
    "encode", "decode", "hash", "sign", "verify", "valid", "invalid", "active", "idle", "ready",
    "failed", "passed", "cached", "synced", "locked", "shared", "atomic", "linear", "binary", "cyclic",
]

SHEET_16_ENG_CONNECTORS: list[str] = [
    "and", "but", "yet", "for", "nor", "so", "if", "then", "else", "elif",
    "when", "while", "until", "since", "unless", "though", "although", "whereas", "thereby", "therefore",
    "thus", "hence", "moreover", "however", "instead", "likewise", "similarly", "finally", "overall", "briefly",
    "note", "tip", "todo", "fixme", "step", "stage", "phase", "case", "rule", "mode",
]

SHEET_17_ENG_PHRASES: list[str] = [
    "at least", "at most", "such as", "in order", "by default", "for example", "as well", "on the other hand",
    "step by step", "end to end", "real time", "low level", "high level", "best case", "worst case", "avg case",
    "time complexity", "space complexity", "base case", "edge case", "key value", "read only", "write once", "up to date",
]

SHEET_18_ENG_PUNCT_MISC: list[str] = [
    "dot", "comma", "colon", "semicolon", "bang", "quest", "dash", "plus", "eq", "tilde",
    "lparen", "rparen", "lbrack", "rbrack", "lbrace", "rbrace", "slash", "bslash", "pipe", "amp",
    "pct", "hash", "at_sign", "star", "caret", "under", "dquote", "squote", "arrow_txt", "eq_arrow",
]

# Sheets 19-22: Computer Science & Engineering Terms
SHEET_19_CS_DBMS_SQL: list[str] = [
    "SELECT", "FROM", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "INSERT", "UPDATE", "DELETE", "JOIN",
    "INNER JOIN", "LEFT JOIN", "PRIMARY KEY", "FOREIGN KEY", "INDEX", "VIEW", "ACID", "ATOM", "CONSIS", "ISOLAT",
    "DURAB", "TRANSACT", "COMMIT", "ROLLBACK", "DEADLOCK", "NORMALIZE", "1NF", "2NF", "3NF", "BCNF",
    "ER_DIAGRAM", "ENTITY", "ATTRIB", "CARDIN", "TUPLE", "SCHEMA", "QUERY", "PLAN", "B_TREE", "HASH_IDX",
]

SHEET_20_CS_NETWORKS: list[str] = [
    "TCP", "UDP", "IP", "IPV4", "IPV6", "HTTP", "HTTPS", "DNS", "DHCP", "ARP",
    "ICMP", "MAC", "SOCKET", "PORT", "HANDSHAKE", "SYN", "ACK", "FIN", "WINDOW", "CONGEST",
    "FLOW_CTRL", "PACKET", "FRAME", "SEGMENT", "DATAGRAM", "SUBNET", "CIDR", "GATEWAY", "ROUTER", "SWITCH",
    "OSI_MODEL", "TCP_IP", "LATENCY", "THROUGHPUT", "BANDWIDTH", "JITTER", "RTT", "TLS", "SSL", "FIREWALL",
]

SHEET_21_CS_OS_DSA: list[str] = [
    "PROCESS", "THREAD", "PCB", "TCB", "SCHED", "FCFS", "SJF", "RR", "PRIORITY", "MUTEX",
    "SEMAPHORE", "PAGING", "PAGE_FAULT", "TLB", "VIRTUAL_MEM", "HEAP", "STACK", "QUEUE", "DEQUE", "TREE",
    "BST", "AVL", "GRAPH", "BFS", "DFS", "DIJKSTRA", "DP", "GREEDY", "RECURSION", "BACKTRACK",
    "BIG_O", "THETA", "OMEGA", "O_1", "O_LOG_N", "O_N", "O_N_LOG_N", "O_N2", "LRU", "FIFO",
]

SHEET_22_ENG_GENERAL_TECH: list[str] = [
    "SYSTEM", "MODEL", "PIPELINE", "INPUT", "OUTPUT", "FEATURE", "LABEL", "LOSS", "EPOCH", "BATCH",
    "TENSOR", "MATRIX", "VECTOR", "DOT_PROD", "NORM", "BIAS", "VARIANCE", "OVERFIT", "UNDERFIT", "METRIC",
    "PRECISION", "RECALL", "F1_SCORE", "AUC", "ROC", "CONF_MAT", "GRAD_CAM", "LORA", "FINETUNE", "INFERENCE",
    "LATEX", "PYTHON", "CPP", "JAVA", "RUST", "GIT", "DOCKER", "API", "JSON", "REST",
]

# Sheets 23-25: Extended Flowchart & Architecture Diagram Sheets
SHEET_23_FLOW_DECISIONS: list[str] = [
    "flow_yes", "flow_no", "flow_true", "flow_false", "flow_loop",
    "flow_retry", "flow_error", "flow_ok", "flow_next", "flow_done",
]

SHEET_24_FLOW_ARCH_NODES: list[str] = [
    "flow_user", "flow_client", "flow_api", "flow_service", "flow_worker",
    "flow_queue", "flow_cache", "flow_db_main", "flow_storage", "flow_ext_svc",
]

SHEET_25_FLOW_CONNECTORS_EX: list[str] = [
    "flow_conn_bend_up", "flow_conn_bend_dn", "flow_conn_merge", "flow_conn_split", "flow_conn_loop_back",
    "flow_conn_dotted", "flow_conn_bold", "flow_conn_label_box", "flow_note_callout", "flow_group_frame",
]

# Sheets 26-45: 20 Additional Templates (English, Math, CS/Engineering, Physics, Biology)
SHEET_26_ENG_8LETTER: list[str] = [
    "function", "variable", "constant", "operator", "argument", "instance", "abstract", "override",
    "protocol", "endpoint", "database", "relation", "schedule", "deadlock", "resource", "allocate",
    "register", "assembly", "compiler", "debugger", "pipeline", "parallel", "document", "template",
    "rendered", "handwritten", "baseline", "sequence", "question", "solution", "analysis", "approach",
    "complete", "required", "optional", "previous", "multiple", "standard", "internal", "external",
]

SHEET_27_ENG_9PLUS_LETTER: list[str] = [
    "algorithm", "complexity", "recursion", "iteration", "structure", "interface", "component", "exception",
    "execution", "condition", "statement", "parameter", "reference", "attribute", "precision", "threshold",
    "parameter", "attention", "transform", "embedding", "inference", "training", "validation", "confusion",
    "therefore", "otherwise", "meanwhile", "furthermore", "specifically", "essentially", "importantly", "accordingly",
]

SHEET_28_ENG_COMMON_PAIRS_2: list[str] = [
    "tr", "pr", "br", "cr", "dr", "fr", "gr", "str", "thr", "chr",
    "pl", "bl", "cl", "fl", "gl", "sl", "sp", "st", "sk", "sm",
    "sn", "sw", "tw", "dw", "qu", "wr", "kn", "gn", "ph", "gh",
    "ai", "au", "aw", "ay", "ea", "ei", "eu", "ew", "ey", "ie",
    "oa", "oe", "oi", "oo", "ou", "ow", "oy", "ue", "ui", "uy",
]

SHEET_29_ENG_COMMON_WORDS_6: list[str] = [
    "people", "before", "little", "should", "around", "another", "between", "through", "because", "against",
    "without", "nothing", "someone", "everyone", "anything", "everything", "sometimes", "together", "different", "important",
    "possible", "actually", "probably", "certainly", "directly", "recently", "quickly", "slowly", "clearly", "simply",
    "already", "instead", "perhaps", "whether", "however", "although", "besides", "towards", "outside", "inside",
]

SHEET_30_ENG_COMMON_WORDS_7: list[str] = [
    "consider", "describe", "explain", "compare", "contrast", "evaluate", "identify", "define", "discuss", "illustrate",
    "prove", "derive", "compute", "observe", "measure", "estimate", "verify", "confirm", "assume", "suppose",
    "obtain", "achieve", "improve", "reduce", "increase", "maintain", "support", "provide", "require", "include",
    "contain", "consist", "perform", "execute", "generate", "produce", "represent", "determine", "indicate", "demonstrate",
]

SHEET_31_ENG_ACADEMIC_NOTES: list[str] = [
    "Definition", "Theorem", "Lemma", "Corollary", "Proof", "Remark", "Example", "Exercise",
    "Case 1", "Case 2", "Case 3", "Step 1", "Step 2", "Step 3", "Given", "To Prove",
    "LHS", "RHS", "Hence", "Thus", "QED", "Note", "Important", "Recall",
    "Summary", "Conclusion", "Observation", "Assumption", "Constraint", "Property", "Result", "Answer",
]

SHEET_32_MATH_LINEAR_ALG: list[str] = [
    "math_rank", "math_trace", "math_span", "math_null", "math_eigen", "math_transpose", "math_inv", "math_norm",
    "math_dot", "math_cross", "math_otimes", "math_oplus", "math_perp_sym", "math_proj", "math_diag", "math_det_A",
    "math_lambda_i", "math_x_vec", "math_A_mat", "math_I_mat", "math_R_n", "math_C_n", "math_Z_set", "math_N_set",
    "math_Q_set", "math_iff", "math_implies", "math_mapsto", "math_subseteq", "math_supseteq", "math_setminus", "math_asymp",
]

SHEET_33_MATH_CALC_STATS: list[str] = [
    "math_d_dx", "math_d2_dx2", "math_partial_x", "math_partial_y", "math_partial_t", "math_lim_x0", "math_lim_inf", "math_sum_i1",
    "math_prod_i1", "math_int_ab", "math_int_0inf", "math_E_X", "math_P_A", "math_P_AB", "math_N_mu_sig", "math_chi2",
    "math_erf", "math_argmax", "math_argmin", "math_sup_x", "math_inf_x", "math_log2", "math_log10", "math_ln_x",
    "math_exp_x", "math_sin_x", "math_cos_x", "math_tan_x", "math_arcsin", "math_arccos", "math_arctan", "math_sinh",
]

SHEET_34_MATH_DISCRETE_LOGIC: list[str] = [
    "math_land", "math_lor", "math_lnot", "math_xor", "math_nand", "math_nor", "math_top", "math_bot",
    "math_vdash", "math_models", "math_equiv_mod", "math_nCr", "math_nPr", "math_floor", "math_ceil", "math_fact",
    "math_aleph0", "math_power_set", "math_big_cup", "math_big_cap", "math_sim", "math_cong", "math_prec", "math_succ",
]

SHEET_35_CS_ALGO_DS: list[str] = [
    "Array", "LinkedList", "DoublyList", "HashMap", "HashSet", "MinHeap", "MaxHeap", "Trie",
    "SegmentTree", "FenwickTree", "DisjointSet", "UnionFind", "AdjList", "AdjMatrix", "TopoSort", "MST",
    "Kruskal", "Prim", "BellmanFord", "FloydWarshall", "QuickSort", "MergeSort", "HeapSort", "BinarySearch",
    "TwoPointer", "SlidingWin", "Memoization", "Tabulation", "Bitmask", "Greedy", "DivideConq", "Amortized",
]

SHEET_36_CS_SYSTEMS_ARCH: list[str] = [
    "CPU", "ALU", "ControlUnit", "Cache_L1", "Cache_L2", "RAM", "Disk", "Bus",
    "Pipeline", "Hazard", "BranchPred", "Interrupt", "SysCall", "Kernel", "UserSpace", "ContextSw",
    "Semaphore", "Monitor", "Spinlock", "Starvation", "Thrashing", "Inode", "FileSystem", "RAID",
    "LoadBalancer", "Microservice", "Container", "Kubernetes", "Sharding", "Replication", "Consensus", "Raft",
]

SHEET_37_CS_AI_ML: list[str] = [
    "Perceptron", "MLP", "CNN", "RNN", "LSTM", "Transformer", "SelfAttn", "MultiHead",
    "Encoder", "Decoder", "Softmax", "ReLU", "GELU", "Sigmoid", "Tanh", "Dropout",
    "BatchNorm", "LayerNorm", "Backprop", "Adam", "SGD", "CrossEntropy", "MSE", "KL_Div",
    "Embedding", "Tokenizer", "Logits", "Epoch", "Overfitting", "Regularize", "Hyperparam", "Checkpoint",
]

SHEET_38_ENG_CIRCUITS_SIGNALS: list[str] = [
    "Voltage", "Current", "Resistance", "Capacitor", "Inductor", "Impedance", "Reactance", "Resonance",
    "Kirchhoff", "KCL", "KVL", "Thevenin", "Norton", "OpAmp", "Diode", "Transistor",
    "MOSFET", "BJT", "CMOS", "FlipFlop", "Mux", "Demux", "Encoder", "Decoder",
    "Fourier", "Laplace", "Z_Transform", "Nyquist", "Sampling", "Bandpass", "Lowpass", "Highpass",
]

SHEET_39_PHYSICS_MECHANICS: list[str] = [
    "Velocity", "Acceleration", "Momentum", "Impulse", "Force", "Torque", "Angular", "Inertia",
    "Kinetic", "Potential", "Work", "Power", "Energy", "Friction", "Gravity", "Oscillation",
    "Amplitude", "Frequency", "Wavelength", "Period", "Pendulum", "Spring", "Hooke", "Elastic",
    "Pressure", "Density", "Buoyancy", "Viscosity", "Bernoulli", "Entropy", "Enthalpy", "Carnot",
]

SHEET_40_PHYSICS_EM_OPTICS: list[str] = [
    "Charge", "Electric", "Magnetic", "Flux", "Gauss", "Faraday", "Lenz", "Maxwell",
    "Coulomb", "Ampere", "BiotSavart", "Solenoid", "Inductance", "Permittivity", "Permeability", "Dielectric",
    "Refraction", "Reflection", "Diffraction", "Interference", "Polarization", "Snell", "FocalLength", "Lens",
    "Photon", "Quantum", "Planck", "DeBroglie", "Photoelectric", "HalfLife", "Isotope", "Fission",
]

SHEET_41_PHYSICS_SYMBOLS: list[str] = [
    "phys_hbar", "phys_angstrom", "phys_ohm", "phys_mu0", "phys_eps0", "phys_c_light", "phys_g_acc", "phys_G_univ",
    "phys_kB", "phys_NA", "phys_R_gas", "phys_e_charge", "phys_me", "phys_mp", "phys_h_planck", "phys_sigma_sb",
    "phys_F_vec", "phys_v_vec", "phys_a_vec", "phys_E_vec", "phys_B_vec", "phys_p_vec", "phys_L_vec", "phys_tau_vec",
]

SHEET_42_BIO_CELL_GENETICS: list[str] = [
    "Cell", "Nucleus", "Mitochondria", "Ribosome", "Chloroplast", "Membrane", "Cytoplasm", "Organelle",
    "Mitosis", "Meiosis", "Chromosome", "Chromatin", "DNA", "RNA", "mRNA", "tRNA",
    "Codon", "Transcription", "Translation", "Replication", "Mutation", "Allele", "Genotype", "Phenotype",
    "Dominant", "Recessive", "Homozygous", "Heterozygous", "Enzyme", "Substrate", "Catalyst", "ATP",
]

SHEET_43_BIO_PHYSIOLOGY: list[str] = [
    "Photosynthesis", "Respiration", "Glycolysis", "KrebsCycle", "Osmosis", "Diffusion", "Homeostasis", "Metabolism",
    "Neuron", "Synapse", "Axon", "Dendrite", "Hormone", "Receptor", "Antibody", "Antigen",
    "Pathogen", "Immunity", "Hemoglobin", "Plasma", "Artery", "Vein", "Capillary", "Alveoli",
    "Nephron", "Ecosystem", "Biodiversity", "Evolution", "Adaptation", "Symbiosis", "Taxonomy", "Species",
]

SHEET_44_ENG_CODE_TOKENS: list[str] = [
    "def", "class", "return", "yield", "async", "await", "import", "from",
    "try", "except", "finally", "raise", "assert", "with", "lambda", "pass",
    "break", "continue", "True", "False", "None", "self", "init", "main",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple",
]

SHEET_45_ENG_FLOW_EXTRAS: list[str] = [
    "flow_start_pill", "flow_end_pill", "flow_if_diamond", "flow_loop_hex", "flow_io_para",
    "flow_db_cyl", "flow_doc_wave", "flow_sub_box", "flow_merge_tri", "flow_note_box",
]

SHEET_CATEGORIES: dict[str, list[int]] = {
    "1. English Sheets (1–5, 12–18, 26–31)": [1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31],
    "2. Math Sheets (6–9, 32–34)": [6, 7, 8, 9, 32, 33, 34],
    "3. Flowchart & Diagram Sheets (10–11, 23–25, 45)": [10, 11, 23, 24, 25, 45],
    "4. CS & Engineering Sheets (19–22, 35–38, 44)": [19, 20, 21, 22, 35, 36, 37, 38, 44],
    "5. Physics & Biology Sheets (39–43)": [39, 40, 41, 42, 43],
}

SHEET_INFO: dict[int, tuple[str, str]] = {
    1: ("Sheet 01 — Single Letters, Digits & Punctuation", "Lowercase a–z, uppercase A–Z, digits 0–9, and punctuation."),
    2: ("Sheet 02 — Double-Letter Joined Pairs", "70 frequent 2-letter pairs (pp, tt, ll, ee, th, er, an, is, on, etc.)."),
    3: ("Sheet 03 — Triple-Letter Common Words & Clusters", "70 frequent 3-letter words and clusters (the, and, ing, ion, for, etc.)."),
    4: ("Sheet 04 — Common 4-Letter Words & Endings", "70 everyday 4-letter words and endings (that, with, have, this, tion, etc.)."),
    5: ("Sheet 05 — Common 5-Letter Words", "70 common 5-letter words (which, their, there, about, would, write, etc.)."),
    6: ("Sheet 06 — Greek Letters (Lowercase & Uppercase)", "Greek letters with clear names."),
    7: ("Sheet 07 — Calculus, Sums & Math Operators", "Integrals, sums, products, inequalities, logic, and set operators."),
    8: ("Sheet 08 — Brackets, Bars & Math Delimiters", "Brackets, braces, slashes, primes, tildes, hats, bars, and dots."),
    9: ("Sheet 09 — Calculus Differentials & Function Names", "dx, dy, dt, sin, cos, tan, log, ln, exp, lim, max, min, det, mod, Var, Cov."),
    10: ("Sheet 10 — Flowchart Nodes & Shapes", "Draw the shape shown in each cell header."),
    11: ("Sheet 11 — Flowchart Arrows & Connectors", "Draw the arrow or connector line shown in each cell header."),
    12: ("Sheet 12 — Common 6-Letter English Words", "High-frequency 6-letter words."),
    13: ("Sheet 13 — Common 7-Letter English Words", "High-frequency 7-letter words."),
    14: ("Sheet 14 — English Prefixes & Suffixes", "Common prefixes and suffixes."),
    15: ("Sheet 15 — Action Verbs & States", "Common verbs and states."),
    16: ("Sheet 16 — Logical Connectors & Markers", "Conjunctions and note markers."),
    17: ("Sheet 17 — Common Multi-Word Phrases", "Frequent two- and three-word phrases."),
    18: ("Sheet 18 — Named Punctuation & Symbol Tokens", "Punctuation and symbol tokens."),
    19: ("Sheet 19 — CS: DBMS & SQL Terms", "SQL clauses, joins, keys, ACID, normalization, and indexing."),
    20: ("Sheet 20 — CS: Computer Networks & Protocols", "TCP/UDP, IP, HTTP/DNS/ARP, handshake, routing, subnetting."),
    21: ("Sheet 21 — CS: OS, DSA & Complexity", "Processes, threads, scheduling, paging, trees/graphs, Big-O, LRU."),
    22: ("Sheet 22 — General Engineering & ML/Dev Terms", "System design, ML metrics, and software engineering terms."),
    23: ("Sheet 23 — Flowchart Branch & Status Labels", "Write the branch/status word shown in each cell header."),
    24: ("Sheet 24 — System & Architecture Diagram Nodes", "Draw/label architecture blocks (Client, API, Queue, Cache, DB, etc.)."),
    25: ("Sheet 25 — Extended Flowchart Connectors & Frames", "Draw the connector, merge/split, or callout shape described."),
    26: ("Sheet 26 — Common 8-Letter English Words", "Frequent 8-letter academic and technical words."),
    27: ("Sheet 27 — Common 9+ Letter English Words", "Longer transition and technical words."),
    28: ("Sheet 28 — Consonant Blends & Vowel Digraphs", "Common English clusters (tr, str, thr, sh, ea, ou, ai, etc.)."),
    29: ("Sheet 29 — Everyday & Academic Adverbs/Pronouns", "Common connecting words (between, through, however, therefore, etc.)."),
    30: ("Sheet 30 — Exam & Problem-Solving Verbs", "Action words used in assignments and exams (derive, prove, compute, etc.)."),
    31: ("Sheet 31 — Lecture & Proof Headings", "Headers like Definition, Theorem, Lemma, Proof, Step 1, LHS, RHS, QED."),
    32: ("Sheet 32 — Math: Linear Algebra & Sets", "rank, trace, span, eigen, transpose, norm, dot, cross, R^n, iff."),
    33: ("Sheet 33 — Math: Calculus & Probability", "d/dx, partial derivatives, limits, expectations E[X], P(A|B), argmax."),
    34: ("Sheet 34 — Math: Discrete Math & Logic", "AND, OR, NOT, XOR, nCr, nPr, floor, ceiling, congruence."),
    35: ("Sheet 35 — CS: Data Structures & Algorithms", "HashMap, MinHeap, Trie, SegmentTree, Dijkstra, DP, BFS, DFS."),
    36: ("Sheet 36 — CS: Computer Architecture & Distributed Systems", "CPU, ALU, Cache, Kernel, Mutex, Sharding, Replication, Raft."),
    37: ("Sheet 37 — CS: Machine Learning & Deep Learning", "Transformer, SelfAttn, Softmax, ReLU, Backprop, Adam, CrossEntropy."),
    38: ("Sheet 38 — Engineering: Circuits, Electronics & Signals", "Voltage, Current, Impedance, KCL, KVL, MOSFET, CMOS, Fourier, Laplace."),
    39: ("Sheet 39 — Physics: Mechanics & Thermodynamics", "Velocity, Acceleration, Momentum, Torque, Work, Power, Entropy."),
    40: ("Sheet 40 — Physics: Electromagnetism, Optics & Modern", "Flux, Gauss, Faraday, Maxwell, Refraction, Diffraction, Quantum."),
    41: ("Sheet 41 — Physics: Constants & Vector Symbols", "hbar, angstrom, ohm, mu_0, eps_0, and vector arrows F, v, a, E, B."),
    42: ("Sheet 42 — Biology: Cell Biology & Genetics", "Nucleus, Mitochondria, DNA, RNA, Mitosis, Meiosis, Codon, Enzyme."),
    43: ("Sheet 43 — Biology: Physiology & Ecology", "Photosynthesis, Respiration, Neuron, Synapse, Antibody, Ecosystem."),
    44: ("Sheet 44 — Programming Keywords & Types", "def, class, return, async, await, import, int, float, str, list, dict."),
    45: ("Sheet 45 — Extra Flowchart & Diagram Shapes", "Additional flowchart pills, diamonds, loops, and callout boxes."),
}

ALL_LIGATURES: list[str] = sorted(
    set(
        SHEET_2_DOUBLE_PAIRS
        + SHEET_3_TRIPLE_WORDS
        + SHEET_4_COMMON_4
        + SHEET_5_COMMON_5
        + [w.removeprefix("lig_") for w in SHEET_9_MATH_COMMON_TERMS]
    ),
    key=len,
    reverse=True,
)

LATEX_MATH_TO_TOKEN: dict[str, str] = {
    r"\alpha": "math_alpha",
    r"\beta": "math_beta",
    r"\gamma": "math_gamma",
    r"\delta": "math_delta",
    r"\Delta": "math_Delta",
    r"\epsilon": "math_epsilon",
    r"\varepsilon": "math_epsilon",
    r"\eta": "math_eta",
    r"\theta": "math_theta",
    r"\Theta": "math_Theta",
    r"\lambda": "math_lambda",
    r"\Lambda": "math_Lambda",
    r"\mu": "math_mu",
    r"\pi": "math_pi",
    r"\rho": "math_rho",
    r"\sigma": "math_sigma",
    r"\Sigma": "math_Sigma",
    r"\tau": "math_tau",
    r"\phi": "math_phi",
    r"\varphi": "math_phi",
    r"\omega": "math_omega",
    r"\Omega": "math_Omega",
    r"\int": "math_int",
    r"\sum": "math_sum",
    r"\prod": "math_prod",
    r"\sqrt": "math_sqrt",
    r"\infty": "math_infty",
    r"\partial": "math_partial",
    r"\nabla": "math_nabla",
    r"\leq": "math_leq",
    r"\le": "math_leq",
    r"\geq": "math_geq",
    r"\ge": "math_geq",
    r"\neq": "math_neq",
    r"\approx": "math_approx",
    r"\times": "math_times",
    r"\cdot": "math_cdot",
    r"\pm": "math_pm",
    r"\rightarrow": "math_rarr",
    r"\to": "math_rarr",
    r"\Rightarrow": "math_Rarr",
    r"\in": "math_in",
    r"\forall": "math_forall",
    r"\exists": "math_exists",
}

MATH_FALLBACK_UNICODE: dict[str, str] = {
    "math_alpha": "α",
    "math_beta": "β",
    "math_gamma": "γ",
    "math_delta": "δ",
    "math_Delta": "Δ",
    "math_epsilon": "ε",
    "math_eta": "η",
    "math_theta": "θ",
    "math_Theta": "Θ",
    "math_lambda": "λ",
    "math_Lambda": "Λ",
    "math_mu": "μ",
    "math_pi": "π",
    "math_rho": "ρ",
    "math_sigma": "σ",
    "math_Sigma": "Σ",
    "math_tau": "τ",
    "math_phi": "φ",
    "math_omega": "ω",
    "math_Omega": "Ω",
    "math_int": "∫",
    "math_sum": "∑",
    "math_prod": "∏",
    "math_sqrt": "√",
    "math_infty": "∞",
    "math_partial": "∂",
    "math_nabla": "∇",
    "math_leq": "≤",
    "math_geq": "≥",
    "math_neq": "≠",
    "math_approx": "≈",
    "math_times": "×",
    "math_cdot": "·",
    "math_pm": "±",
    "math_rarr": "→",
    "math_Rarr": "⇒",
    "math_in": "∈",
    "math_forall": "∀",
    "math_exists": "∃",
}

MATH_SYMBOL_LABELS: dict[str, str] = {
    "math_alpha": "α (alpha)",
    "math_beta": "β (beta)",
    "math_gamma": "γ (gamma)",
    "math_delta": "δ (delta)",
    "math_epsilon": "ε (eps)",
    "math_zeta": "ζ (zeta)",
    "math_eta": "η (eta)",
    "math_theta": "θ (theta)",
    "math_iota": "ι (iota)",
    "math_kappa": "κ (kappa)",
    "math_lambda": "λ (lambda)",
    "math_mu": "μ (mu)",
    "math_nu": "ν (nu)",
    "math_xi": "ξ (xi)",
    "math_pi": "π (pi)",
    "math_rho": "ρ (rho)",
    "math_sigma": "σ (sigma)",
    "math_tau": "τ (tau)",
    "math_upsilon": "υ (ups)",
    "math_phi": "φ (phi)",
    "math_chi": "χ (chi)",
    "math_psi": "ψ (psi)",
    "math_omega": "ω (omega)",
    "math_Gamma": "Γ (Gamma)",
    "math_Delta": "Δ (Delta)",
    "math_Theta": "Θ (Theta)",
    "math_Lambda": "Λ (Lambda)",
    "math_Xi": "Ξ (Xi)",
    "math_Pi": "Π (Pi)",
    "math_Sigma": "Σ (Sigma)",
    "math_Phi": "Φ (Phi)",
    "math_Psi": "Ψ (Psi)",
    "math_Omega": "Ω (Omega)",
    "math_int": "∫ (int)",
    "math_iint": "∬ (iint)",
    "math_oint": "∮ (oint)",
    "math_sum": "∑ (sum)",
    "math_prod": "∏ (prod)",
    "math_sqrt": "√ (sqrt)",
    "math_partial": "∂ (part)",
    "math_nabla": "∇ (nabla)",
    "math_infty": "∞ (inf)",
    "math_pm": "± (pm)",
    "math_mp": "∓ (mp)",
    "math_times": "× (times)",
    "math_div": "÷ (div)",
    "math_cdot": "· (cdot)",
    "math_approx": "≈ (approx)",
    "math_neq": "≠ (neq)",
    "math_equiv": "≡ (equiv)",
    "math_leq": "≤ (leq)",
    "math_geq": "≥ (geq)",
    "math_ll": "≪ (ll)",
    "math_gg": "≫ (gg)",
    "math_propto": "∝ (prop)",
    "math_rarr": "→ (rarr)",
    "math_larr": "← (larr)",
    "math_harr": "↔ (harr)",
    "math_Rarr": "⇒ (Rarr)",
    "math_Harr": "⇔ (Harr)",
    "math_forall": "∀ (forall)",
    "math_exists": "∃ (exists)",
    "math_in": "∈ (in)",
    "math_notin": "∉ (notin)",
    "math_subset": "⊂ (sub)",
    "math_cup": "∪ (cup)",
    "math_cap": "∩ (cap)",
    "math_emptyset": "∅ (empty)",
    "math_therefore": "∴ (theref)",
    "math_because": "∵ (bec)",
    "lparen": "(",
    "rparen": ")",
    "lbrack": "[",
    "rbrack": "]",
    "lbrace": "{",
    "rbrace": "}",
    "vbar": "|",
    "dblbar": "‖",
    "langle": "⟨",
    "rangle": "⟩",
    "slash": "/",
    "bslash": "\\",
    "caret": "^",
    "underscore": "_",
    "degree": "°",
    "prime": "′",
    "dprime": "″",
    "angle": "∠",
    "perp": "⊥",
    "parallel": "∥",
    "triangle": "△",
    "square_op": "□",
    "tilde_op": "~",
    "hat_op": "^ (hat)",
    "bar_op": "¯ (bar)",
    "vec_op": "⃗ (vec)",
    "dot_op": "˙ (dot)",
    "ddot_op": "¨ (ddot)",
    "star_op": "*",
    "circ_op": "∘",
    "flow_oval": "Oval (Start)",
    "flow_rect": "Rect (Step)",
    "flow_diamond": "Diamond (?)",
    "flow_parallelogram": "Parallelogram",
    "flow_circle": "Circle (Conn)",
    "flow_cylinder": "Cylinder (DB)",
    "flow_hexagon": "Hexagon",
    "flow_document": "Document",
    "flow_cloud": "Cloud",
    "flow_subproc": "Subprocess",
    "flow_arr_right": "Arrow →",
    "flow_arr_left": "Arrow ←",
    "flow_arr_down": "Arrow ↓",
    "flow_arr_up": "Arrow ↑",
    "flow_arr_bi": "Arrow ↔",
    "flow_elbow_rd": "Elbow ↳",
    "flow_elbow_dr": "Elbow ↴",
    "flow_arr_dashed": "Dashed ⇢",
    "flow_line_h": "Line —",
    "flow_line_v": "Line |",
}

CHAR_TO_SPECIAL_FOLDER: dict[str, str] = {
    '"': "quote_double",
    "'": "quote_single",
    "?": "question",
    ":": "colon",
    ";": "semicolon",
    "(": "lparen",
    ")": "rparen",
    "[": "lbrack",
    "]": "rbrack",
    "{": "lbrace",
    "}": "rbrace",
    "|": "vbar",
    "/": "slash",
    "\\": "bslash",
    "^": "caret",
    "_": "underscore",
    "°": "degree",
    "*": "star_op",
    "~": "tilde_op",
}

UNICODE_TO_MATH_FOLDER: dict[str, str] = {v: k for k, v in MATH_FALLBACK_UNICODE.items()}
TALL_BRACKETS = set("()[]{}|")

BUILTIN_PRESETS: dict[str, dict[str, object]] = {
    "Black Pen (Ruled White)": {
        "bg_mode": "Ruled White",
        "default_hex": "#181A1F",
        "opacity": 1.00,
        "stroke_delta": 0,
        "base_size": 30,
        "cap_scale": 1.06,
        "lower_scale": 0.53,
        "width_scale": 0.86,
        "line_stretch": 0.92,
        "script_scale": 0.75,
        "script_height_offset": 0.25,
        "char_spacing": -2,
        "word_spacing": 12,
        "line_gap": 64,
        "margin_side": 85,
        "baseline_jitter": 1,
        "tilt_deg": 0.7,
    },
    "Balanced Notebook (Default)": {
        "bg_mode": "Plain White",
        "default_hex": "#1C1E24",
        "opacity": 1.00,
        "stroke_delta": 0,
        "base_size": 33,
        "cap_scale": 1.00,
        "lower_scale": 0.48,
        "width_scale": 1.00,
        "line_stretch": 1.00,
        "script_scale": 0.72,
        "script_height_offset": 0.25,
        "char_spacing": -2,
        "word_spacing": 19,
        "line_gap": 64,
        "margin_side": 115,
        "baseline_jitter": 1,
        "tilt_deg": 0.8,
    },
    "Black Pen (Beige Ruled)": {
        "bg_mode": "Beige Ruled",
        "default_hex": "#1A1C22",
        "opacity": 1.00,
        "stroke_delta": 0,
        "base_size": 34,
        "cap_scale": 1.00,
        "lower_scale": 0.50,
        "width_scale": 1.04,
        "line_stretch": 1.04,
        "script_scale": 0.74,
        "script_height_offset": 0.25,
        "char_spacing": -1,
        "word_spacing": 20,
        "line_gap": 64,
        "margin_side": 105,
        "baseline_jitter": 1,
        "tilt_deg": 0.7,
    },
    "Blue Fountain Pen (Ruled)": {
        "bg_mode": "Ruled White",
        "default_hex": "#1C3E94",
        "opacity": 1.00,
        "stroke_delta": 1,
        "base_size": 33,
        "cap_scale": 1.05,
        "lower_scale": 0.50,
        "width_scale": 1.02,
        "line_stretch": 1.02,
        "script_scale": 0.72,
        "script_height_offset": 0.25,
        "char_spacing": -1,
        "word_spacing": 20,
        "line_gap": 64,
        "margin_side": 115,
        "baseline_jitter": 1,
        "tilt_deg": 1.0,
    },
    "Blue Gel Pen (Plain White)": {
        "bg_mode": "Plain White",
        "default_hex": "#19388A",
        "opacity": 1.00,
        "stroke_delta": 1,
        "base_size": 36,
        "cap_scale": 1.04,
        "lower_scale": 0.52,
        "width_scale": 1.06,
        "line_stretch": 1.05,
        "script_scale": 0.75,
        "script_height_offset": 0.26,
        "char_spacing": -1,
        "word_spacing": 22,
        "line_gap": 64,
        "margin_side": 105,
        "baseline_jitter": 1,
        "tilt_deg": 0.8,
    },
    "Compact Study Notes (Beige)": {
        "bg_mode": "Beige Ruled",
        "default_hex": "#1C1E24",
        "opacity": 1.00,
        "stroke_delta": 0,
        "base_size": 33,
        "cap_scale": 1.00,
        "lower_scale": 0.48,
        "width_scale": 0.98,
        "line_stretch": 1.00,
        "script_scale": 0.70,
        "script_height_offset": 0.24,
        "char_spacing": -3,
        "word_spacing": 12,
        "line_gap": 64,
        "margin_side": 95,
        "baseline_jitter": 1,
        "tilt_deg": 0.6,
    },
    "Exam Answer Sheet (Wide Ruled)": {
        "bg_mode": "Ruled White",
        "default_hex": "#1B263B",
        "opacity": 1.00,
        "stroke_delta": 0,
        "base_size": 38,
        "cap_scale": 1.06,
        "lower_scale": 0.54,
        "width_scale": 1.08,
        "line_stretch": 1.08,
        "script_scale": 0.76,
        "script_height_offset": 0.26,
        "char_spacing": 0,
        "word_spacing": 24,
        "line_gap": 68,
        "margin_side": 100,
        "baseline_jitter": 1,
        "tilt_deg": 0.7,
    },
}


def load_saved_presets() -> dict[str, dict[str, object]]:
    merged = {k: dict(v) for k, v in BUILTIN_PRESETS.items()}
    if PRESETS_FILE.exists():
        try:
            disk_data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(disk_data, dict):
                for name, cfg in disk_data.items():
                    if isinstance(cfg, dict):
                        merged[str(name)] = cfg
        except Exception:
            pass
    return merged


def persist_custom_preset(name: str, cfg: dict[str, object]) -> None:
    custom_only: dict[str, dict[str, object]] = {}
    if PRESETS_FILE.exists():
        try:
            existing = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                custom_only = existing
        except Exception:
            custom_only = {}
    custom_only[name] = cfg
    PRESETS_FILE.write_text(json.dumps(custom_only, indent=2), encoding="utf-8")


def load_crisp_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (28, 30, 36)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


SAFE_TEMPLATE_LABELS: dict[str, str] = {
    "math_iint": "iint (double int)",
    "math_oint": "oint (contour int)",
    "math_nabla": "nabla (del)",
    "math_pm": "+/- (pm)",
    "math_mp": "-/+ (mp)",
    "math_ll": "<< (much less)",
    "math_gg": ">> (much greater)",
    "math_propto": "prop (propto)",
    "math_rarr": "-> (rarr)",
    "math_larr": "<- (larr)",
    "math_harr": "<-> (harr)",
    "math_Rarr": "=> (Rarr)",
    "math_Harr": "<=> (Harr)",
    "math_forall": "forall (A-inv)",
    "math_exists": "exists (E-rev)",
    "math_in": "in (elem)",
    "math_notin": "notin",
    "math_subset": "subset (sub)",
    "math_cup": "cup (union)",
    "math_cap": "cap (inter)",
    "math_emptyset": "emptyset",
    "math_therefore": "therefore (.:)",
    "math_because": "because (:. )",
    "dblbar": "|| (dblbar)",
    "langle": "< (langle)",
    "rangle": "> (rangle)",
    "degree": "deg (circ)",
    "prime": "prime (')",
    "dprime": "dprime ('')",
    "angle": "angle (<)",
    "perp": "perp (_|_)",
    "parallel": "parallel (||)",
    "triangle": "triangle (/\\)",
    "square_op": "square ([])",
    "hat_op": "hat (^)",
    "bar_op": "bar (-)",
    "vec_op": "vec (->)",
    "dot_op": "dot (.)",
    "ddot_op": "ddot (..)",
    "circ_op": "circ (o)",
    "flow_elbow_rd": "Elbow R->D",
    "flow_elbow_dr": "Elbow D->R",
    "flow_arr_dashed": "Dashed --->",
}


EXPLICIT_CELL_GUIDES: dict[str, str] = {
    # Flowchart nodes & arrows (Draw instructions)
    "flow_oval": "Draw: Oval (Start/End)",
    "flow_rect": "Draw: Rectangle (Step)",
    "flow_diamond": "Draw: Diamond (If ?)",
    "flow_parallelogram": "Draw: Slanted Box (I/O)",
    "flow_circle": "Draw: Small Circle",
    "flow_cylinder": "Draw: Cylinder (DB)",
    "flow_hexagon": "Draw: Hexagon (Prep)",
    "flow_document": "Draw: Wavy-Base Doc",
    "flow_cloud": "Draw: Cloud Shape",
    "flow_subproc": "Draw: Double-Side Box",
    "flow_arr_right": "Draw: Arrow ->",
    "flow_arr_left": "Draw: Arrow <-",
    "flow_arr_down": "Draw: Arrow Down",
    "flow_arr_up": "Draw: Arrow Up",
    "flow_arr_bi": "Draw: Arrow <->",
    "flow_elbow_rd": "Draw: Elbow Right->Down",
    "flow_elbow_dr": "Draw: Elbow Down->Right",
    "flow_arr_dashed": "Draw: Dashed Arrow --->",
    "flow_line_h": "Draw: Horizontal Line —",
    "flow_line_v": "Draw: Vertical Line |",
    # Flowchart branch labels (Write instructions)
    "flow_yes": "Write: Yes",
    "flow_no": "Write: No",
    "flow_true": "Write: True",
    "flow_false": "Write: False",
    "flow_loop": "Write: Loop",
    "flow_retry": "Write: Retry",
    "flow_error": "Write: Error",
    "flow_ok": "Write: OK",
    "flow_next": "Write: Next",
    "flow_done": "Write: Done",
    # Architecture nodes (Draw + Label)
    "flow_user": "Draw: Stick User",
    "flow_client": "Box: Client",
    "flow_api": "Box: API",
    "flow_service": "Box: Service",
    "flow_worker": "Box: Worker",
    "flow_queue": "Draw: Pipe/Queue",
    "flow_cache": "Box: Cache",
    "flow_db_main": "Draw: DB Cylinder",
    "flow_storage": "Draw: Bucket/Disk",
    "flow_ext_svc": "Draw: Cloud Svc",
    # Extended connectors
    "flow_conn_bend_up": "Draw: Bend Up _/",
    "flow_conn_bend_dn": "Draw: Bend Down \\_",
    "flow_conn_merge": "Draw: Merge >-",
    "flow_conn_split": "Draw: Split -<",
    "flow_conn_loop_back": "Draw: Loop-Back U-Turn",
    "flow_conn_dotted": "Draw: Dotted Line ...",
    "flow_conn_bold": "Draw: Thick Line ===",
    "flow_conn_label_box": "Draw: Small Tag Box",
    "flow_note_callout": "Draw: Note Callout",
    "flow_group_frame": "Draw: Dashed Frame",
    "flow_start_pill": "Draw: Pill (Start)",
    "flow_end_pill": "Draw: Pill (End)",
    "flow_if_diamond": "Draw: Diamond (?)",
    "flow_loop_hex": "Draw: Hexagon Loop",
    "flow_io_para": "Draw: Parallelogram",
    "flow_db_cyl": "Draw: Cylinder DB",
    "flow_doc_wave": "Draw: Document Box",
    "flow_sub_box": "Draw: Subprocess Box",
    "flow_merge_tri": "Draw: Triangle Merge",
    "flow_note_box": "Draw: Folded Note",
    # Math & Physics explicit guides
    "math_iint": "Draw: Double Integral",
    "math_oint": "Draw: Loop Integral",
    "math_nabla": "Draw: Inverted Triangle",
    "math_pm": "Draw: + over -",
    "math_mp": "Draw: - over +",
    "math_ll": "Draw: <<",
    "math_gg": "Draw: >>",
    "math_propto": "Draw: Proportional",
    "math_rarr": "Draw: ->",
    "math_larr": "Draw: <-",
    "math_harr": "Draw: <->",
    "math_Rarr": "Draw: =>",
    "math_Harr": "Draw: <=>",
    "math_forall": "Draw: Upside-Down A",
    "math_exists": "Draw: Backwards E",
    "math_in": "Draw: Element-Of (E)",
    "math_notin": "Draw: Slashed E",
    "math_subset": "Draw: Subset C",
    "math_cup": "Draw: Union U",
    "math_cap": "Draw: Intersection",
    "math_emptyset": "Draw: Slashed Zero",
    "math_therefore": "Draw: 3 Dots (Up)",
    "math_because": "Draw: 3 Dots (Down)",
    "phys_hbar": "Write: h-bar",
    "phys_angstrom": "Write: A with ring",
    "phys_ohm": "Draw: Omega (Ohm)",
    "phys_mu0": "Write: mu_0",
    "phys_eps0": "Write: eps_0",
    "phys_c_light": "Write: c",
    "phys_g_acc": "Write: g",
    "phys_G_univ": "Write: G",
    "phys_kB": "Write: k_B",
    "phys_NA": "Write: N_A",
    "phys_R_gas": "Write: R",
    "phys_e_charge": "Write: e",
    "phys_me": "Write: m_e",
    "phys_mp": "Write: m_p",
    "phys_h_planck": "Write: h",
    "phys_sigma_sb": "Write: sigma",
    "phys_F_vec": "Write: F with arrow",
    "phys_v_vec": "Write: v with arrow",
    "phys_a_vec": "Write: a with arrow",
    "phys_E_vec": "Write: E with arrow",
    "phys_B_vec": "Write: B with arrow",
    "phys_p_vec": "Write: p with arrow",
    "phys_L_vec": "Write: L with arrow",
    "phys_tau_vec": "Write: tau with arrow",
}



DRAW_CELL_GUIDES: dict[str, str] = {
    # Sheet 10 & 45: Flowchart Nodes & Shapes
    "flow_oval": "Draw: Oval",
    "flow_rect": "Draw: Rectangle",
    "flow_diamond": "Draw: Diamond",
    "flow_parallelogram": "Draw: Parallelogram",
    "flow_circle": "Draw: Circle",
    "flow_cylinder": "Draw: Cylinder",
    "flow_hexagon": "Draw: Hexagon",
    "flow_document": "Draw: Document Box",
    "flow_cloud": "Draw: Cloud",
    "flow_subproc": "Draw: Subprocess Box",
    "flow_start_pill": "Draw: Pill (Start)",
    "flow_end_pill": "Draw: Pill (End)",
    "flow_if_diamond": "Draw: Diamond (?)",
    "flow_loop_hex": "Draw: Hexagon",
    "flow_io_para": "Draw: Parallelogram",
    "flow_db_cyl": "Draw: Cylinder",
    "flow_doc_wave": "Draw: Wavy Document",
    "flow_sub_box": "Draw: Double-Side Box",
    "flow_merge_tri": "Draw: Triangle",
    "flow_note_box": "Draw: Folded Note",
    # Sheet 11 & 25: Flowchart Arrows & Connectors
    "flow_arr_right": "Draw: Arrow ->",
    "flow_arr_left": "Draw: Arrow <-",
    "flow_arr_down": "Draw: Down Arrow",
    "flow_arr_up": "Draw: Up Arrow",
    "flow_arr_bi": "Draw: Arrow <->",
    "flow_elbow_rd": "Draw: Elbow Right->Down",
    "flow_elbow_dr": "Draw: Elbow Down->Right",
    "flow_arr_dashed": "Draw: Dashed Arrow --->",
    "flow_line_h": "Draw: Horizontal Line",
    "flow_line_v": "Draw: Vertical Line",
    "flow_conn_bend_up": "Draw: Bend Up",
    "flow_conn_bend_dn": "Draw: Bend Down",
    "flow_conn_merge": "Draw: Merge Line",
    "flow_conn_split": "Draw: Split Line",
    "flow_conn_loop_back": "Draw: Loop-Back Arrow",
    "flow_conn_dotted": "Draw: Dotted Line ...",
    "flow_conn_bold": "Draw: Thick Line ===",
    "flow_conn_label_box": "Draw: Tag Box",
    "flow_note_callout": "Draw: Callout Box",
    "flow_group_frame": "Draw: Dashed Frame",
    # Sheet 24: Architecture Diagram Nodes
    "flow_user": "Draw: Stick Person",
    "flow_client": "Draw: Client Box",
    "flow_api": "Draw: API Box",
    "flow_service": "Draw: Service Box",
    "flow_worker": "Draw: Worker Box",
    "flow_queue": "Draw: Queue Pipe",
    "flow_cache": "Draw: Cache Box",
    "flow_db_main": "Draw: DB Cylinder",
    "flow_storage": "Draw: Storage Disk",
    "flow_ext_svc": "Draw: Cloud Service",
    # Sheet 8: Geometric Shapes
    "triangle": "Draw: Triangle",
    "square_op": "Draw: Square",
    "angle": "Draw: Angle Sign",
    "perp": "Draw: Perp Sign",
    "parallel": "Draw: Parallel ||",
}

WRITE_CELL_GUIDES: dict[str, str] = {
    # Sheet 1: Punctuation & Special
    "question": "Write: ?",
    "quote_single": "Write: '",
    "quote_double": 'Write: "',
    "colon": "Write: :",
    "semicolon": "Write: ;",
    # Sheet 6: Greek Letters (universal symbols + names)
    "math_alpha": "Write: α (alpha)",
    "math_beta": "Write: β (beta)",
    "math_gamma": "Write: γ (gamma)",
    "math_delta": "Write: δ (delta)",
    "math_epsilon": "Write: ε (epsilon)",
    "math_zeta": "Write: ζ (zeta)",
    "math_eta": "Write: η (eta)",
    "math_theta": "Write: θ (theta)",
    "math_iota": "Write: ι (iota)",
    "math_kappa": "Write: κ (kappa)",
    "math_lambda": "Write: λ (lambda)",
    "math_mu": "Write: μ (mu)",
    "math_nu": "Write: ν (nu)",
    "math_xi": "Write: ξ (xi)",
    "math_pi": "Write: π (pi)",
    "math_rho": "Write: ρ (rho)",
    "math_sigma": "Write: σ (sigma)",
    "math_tau": "Write: τ (tau)",
    "math_upsilon": "Write: υ (upsilon)",
    "math_phi": "Write: φ (phi)",
    "math_chi": "Write: χ (chi)",
    "math_psi": "Write: ψ (psi)",
    "math_omega": "Write: ω (omega)",
    "math_Gamma": "Write: Γ (Gamma)",
    "math_Delta": "Write: Δ (Delta)",
    "math_Theta": "Write: Θ (Theta)",
    "math_Lambda": "Write: Λ (Lambda)",
    "math_Xi": "Write: Ξ (Xi)",
    "math_Pi": "Write: Π (Pi)",
    "math_Sigma": "Write: Σ (Sigma)",
    "math_Phi": "Write: Φ (Phi)",
    "math_Psi": "Write: Ψ (Psi)",
    "math_Omega": "Write: Ω (Omega)",
    # Sheet 7: Math Operators (no broken box glyphs)
    "math_int": "Write: ∫ (integral)",
    "math_iint": "Write: double integral",
    "math_oint": "Write: contour integral",
    "math_sum": "Write: ∑ (sum)",
    "math_prod": "Write: ∏ (product)",
    "math_sqrt": "Write: √ (sqrt)",
    "math_partial": "Write: ∂ (partial)",
    "math_nabla": "Write: nabla (del)",
    "math_infty": "Write: ∞ (infinity)",
    "math_pm": "Write: ± (plus-minus)",
    "math_mp": "Write: minus-plus",
    "math_times": "Write: × (times)",
    "math_div": "Write: ÷ (divide)",
    "math_cdot": "Write: · (dot)",
    "math_approx": "Write: ≈ (approx)",
    "math_neq": "Write: ≠ (not equal)",
    "math_equiv": "Write: ≡ (equiv)",
    "math_leq": "Write: ≤ (leq)",
    "math_geq": "Write: ≥ (geq)",
    "math_ll": "Write: <<",
    "math_gg": "Write: >>",
    "math_propto": "Write: proportional",
    "math_rarr": "Write: ->",
    "math_larr": "Write: <-",
    "math_harr": "Write: <->",
    "math_Rarr": "Write: =>",
    "math_Harr": "Write: <=>",
    "math_forall": "Write: for all",
    "math_exists": "Write: there exists",
    "math_in": "Write: element of",
    "math_notin": "Write: not in",
    "math_subset": "Write: subset",
    "math_cup": "Write: union (U)",
    "math_cap": "Write: intersection",
    "math_emptyset": "Write: empty set",
    "math_therefore": "Write: therefore",
    "math_because": "Write: because",
    # Sheet 8: Brackets & Delimiters
    "lparen": "Write: (",
    "rparen": "Write: )",
    "lbrack": "Write: [",
    "rbrack": "Write: ]",
    "lbrace": "Write: {",
    "rbrace": "Write: }",
    "vbar": "Write: |",
    "dblbar": "Write: ||",
    "langle": "Write: <",
    "rangle": "Write: >",
    "slash": "Write: /",
    "bslash": "Write: backslash",
    "caret": "Write: ^",
    "underscore": "Write: _",
    "degree": "Write: ° (degree)",
    "prime": "Write: ' (prime)",
    "dprime": "Write: '' (double prime)",
    "tilde_op": "Write: ~",
    "hat_op": "Write: ^ (hat)",
    "bar_op": "Write: - (bar)",
    "vec_op": "Write: -> (vec)",
    "dot_op": "Write: . (dot)",
    "ddot_op": "Write: .. (ddot)",
    "star_op": "Write: *",
    "circ_op": "Write: o (comp)",
    # Sheet 18: Named Punctuation & Symbol Tokens
    "dot": "Write: .",
    "comma": "Write: ,",
    "bang": "Write: !",
    "quest": "Write: ?",
    "dash": "Write: -",
    "plus": "Write: +",
    "eq": "Write: =",
    "tilde": "Write: ~",
    "pipe": "Write: |",
    "amp": "Write: &",
    "pct": "Write: %",
    "hash": "Write: #",
    "at_sign": "Write: @",
    "star": "Write: *",
    "under": "Write: _",
    "dquote": 'Write: "',
    "squote": "Write: '",
    "arrow_txt": "Write: ->",
    "eq_arrow": "Write: =>",
    # Sheet 23: Flowchart Branch Words
    "flow_yes": "Write: Yes",
    "flow_no": "Write: No",
    "flow_true": "Write: True",
    "flow_false": "Write: False",
    "flow_loop": "Write: Loop",
    "flow_retry": "Write: Retry",
    "flow_error": "Write: Error",
    "flow_ok": "Write: OK",
    "flow_next": "Write: Next",
    "flow_done": "Write: Done",
    # Sheets 32–34: Math Linear Algebra, Calculus, Logic
    "math_d_dx": "Write: d/dx",
    "math_d2_dx2": "Write: d2/dx2",
    "math_partial_x": "Write: ∂/∂x",
    "math_partial_y": "Write: ∂/∂y",
    "math_partial_t": "Write: ∂/∂t",
    "math_lim_x0": "Write: lim x->0",
    "math_lim_inf": "Write: lim x->∞",
    "math_sum_i1": "Write: ∑ i=1",
    "math_prod_i1": "Write: ∏ i=1",
    "math_int_ab": "Write: ∫ a to b",
    "math_int_0inf": "Write: ∫ 0 to ∞",
    "math_E_X": "Write: E[X]",
    "math_P_A": "Write: P(A)",
    "math_P_AB": "Write: P(A|B)",
    "math_N_mu_sig": "Write: N(μ, σ)",
    "math_chi2": "Write: χ^2",
    "math_erf": "Write: erf",
    "math_argmax": "Write: argmax",
    "math_argmin": "Write: argmin",
    "math_sup_x": "Write: sup",
    "math_inf_x": "Write: inf",
    "math_log2": "Write: log2",
    "math_log10": "Write: log10",
    "math_ln_x": "Write: ln(x)",
    "math_exp_x": "Write: exp(x)",
    "math_sin_x": "Write: sin(x)",
    "math_cos_x": "Write: cos(x)",
    "math_tan_x": "Write: tan(x)",
    "math_arcsin": "Write: arcsin",
    "math_arccos": "Write: arccos",
    "math_arctan": "Write: arctan",
    "math_sinh": "Write: sinh",
    "math_transpose": "Write: A^T",
    "math_inv": "Write: A^-1",
    "math_norm": "Write: ||x||",
    "math_dot": "Write: a · b",
    "math_cross": "Write: a × b",
    "math_otimes": "Write: tensor prod",
    "math_oplus": "Write: direct sum",
    "math_perp_sym": "Write: perp",
    "math_det_A": "Write: det(A)",
    "math_lambda_i": "Write: λ_i",
    "math_x_vec": "Write: x (vec)",
    "math_A_mat": "Write: A",
    "math_I_mat": "Write: I",
    "math_R_n": "Write: R^n",
    "math_C_n": "Write: C^n",
    "math_Z_set": "Write: Z",
    "math_N_set": "Write: N",
    "math_Q_set": "Write: Q",
    "math_iff": "Write: iff",
    "math_implies": "Write: =>",
    "math_mapsto": "Write: |->",
    "math_subseteq": "Write: subseteq",
    "math_supseteq": "Write: supseteq",
    "math_setminus": "Write:  (backslash)",
    "math_asymp": "Write: ~",
    "math_land": "Write: AND",
    "math_lor": "Write: OR",
    "math_lnot": "Write: NOT",
    "math_xor": "Write: XOR",
    "math_nand": "Write: NAND",
    "math_nor": "Write: NOR",
    "math_top": "Write: True (T)",
    "math_bot": "Write: False (F)",
    "math_vdash": "Write: proves",
    "math_models": "Write: models",
    "math_equiv_mod": "Write: ≡ (mod)",
    "math_nCr": "Write: nCr",
    "math_nPr": "Write: nPr",
    "math_floor": "Write: floor(x)",
    "math_ceil": "Write: ceil(x)",
    "math_fact": "Write: n!",
    "math_aleph0": "Write: aleph_0",
    "math_power_set": "Write: P(S)",
    "math_big_cup": "Write: Union",
    "math_big_cap": "Write: Intersect",
    "math_sim": "Write: ~",
    "math_cong": "Write: congruent",
    "math_prec": "Write: precedes",
    "math_succ": "Write: succeeds",
    # Sheet 41: Physics Symbols & Vectors
    "phys_hbar": "Write: h-bar",
    "phys_angstrom": "Write: Angstrom",
    "phys_ohm": "Write: Ω (Ohm)",
    "phys_mu0": "Write: μ_0",
    "phys_eps0": "Write: ε_0",
    "phys_c_light": "Write: c",
    "phys_g_acc": "Write: g",
    "phys_G_univ": "Write: G",
    "phys_kB": "Write: k_B",
    "phys_NA": "Write: N_A",
    "phys_R_gas": "Write: R",
    "phys_e_charge": "Write: e",
    "phys_me": "Write: m_e",
    "phys_mp": "Write: m_p",
    "phys_h_planck": "Write: h",
    "phys_sigma_sb": "Write: σ",
    "phys_F_vec": "Write: F (vec)",
    "phys_v_vec": "Write: v (vec)",
    "phys_a_vec": "Write: a (vec)",
    "phys_E_vec": "Write: E (vec)",
    "phys_B_vec": "Write: B (vec)",
    "phys_p_vec": "Write: p (vec)",
    "phys_L_vec": "Write: L (vec)",
    "phys_tau_vec": "Write: τ (vec)",
}


def display_label(folder_name: str) -> str:
    if folder_name in DRAW_CELL_GUIDES:
        return DRAW_CELL_GUIDES[folder_name]
    if folder_name in WRITE_CELL_GUIDES:
        return WRITE_CELL_GUIDES[folder_name]
    if folder_name.startswith("upper_"):
        return f"Write: {folder_name.removeprefix('upper_').upper()} (cap)"
    for pfx in (
        "cs_db_", "cs_net_", "cs_os_", "cs_tech_", "cs_algo_", "cs_sys_", "cs_ai_",
        "eng_ee_", "phys_", "bio_", "code_", "word_", "affix_", "phrase_", "lig_", "math_", "flow_",
    ):
        if folder_name.startswith(pfx):
            cleaned = folder_name.removeprefix(pfx).replace("_", " ").strip()
            return f"Write: {cleaned}"
    return f"Write: {folder_name}"


def char_to_folder(ch: str) -> str | None:
    if ch in CHAR_TO_SPECIAL_FOLDER:
        return CHAR_TO_SPECIAL_FOLDER[ch]
    if ch in UNICODE_TO_MATH_FOLDER:
        return UNICODE_TO_MATH_FOLDER[ch]
    if "A" <= ch <= "Z":
        return f"upper_{ch}"
    if "a" <= ch <= "z" or ch in "0123456789.,-+=!:;":
        return ch
    return None


def tokenize_word_with_ligatures(word: str, bank: dict[str, list[Image.Image]]) -> list[tuple[str, str | None]]:
    tokens: list[tuple[str, str | None]] = []
    i = 0
    n = len(word)

    while i < n:
        matched_lig: str | None = None
        for lig in ALL_LIGATURES:
            end = i + len(lig)
            if end <= n and word[i:end] == lig:
                folder = f"lig_{lig}"
                if bank.get(folder):
                    matched_lig = lig
                    tokens.append((lig, folder))
                    i = end
                    break
        if matched_lig is None:
            ch = word[i]
            m_folder = char_to_folder(ch)
            tokens.append((ch, m_folder))
            i += 1
    return tokens


def extract_braced_arg(s: str, start_idx: int) -> tuple[str, int]:
    """Given s[start_idx] == '{', return (inner_text, index_after_matching_brace)."""
    if start_idx >= len(s) or s[start_idx] != "{":
        return s[start_idx : start_idx + 1], start_idx + 1
    depth = 1
    i = start_idx + 1
    while i < len(s) and depth > 0:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
        i += 1
    inner = s[start_idx + 1 : i - 1] if depth == 0 else s[start_idx + 1 : i]
    return inner, i


def token_height_multiplier(token: str, lower_scale: float = 0.48, cap_scale: float = 1.00) -> float:
    x_height = max(lower_scale * 1.28, 0.60)
    asc_desc_height = max(x_height * 1.52, lower_scale * 1.88, 0.88)
    full_loop_height = max(cap_scale * 1.22, asc_desc_height * 1.32)

    if token in UNICODE_TO_MATH_FOLDER:
        folder = UNICODE_TO_MATH_FOLDER[token]
        if folder in (
            "math_eta", "math_theta", "math_int", "math_sum", "math_prod", "math_sqrt",
            "math_Gamma", "math_Delta", "math_Theta", "math_Lambda",
            "math_Xi", "math_Pi", "math_Sigma", "math_Phi", "math_Psi", "math_Omega",
        ):
            return max(cap_scale * 1.02, 0.94)
        return max(cap_scale * 0.88, 0.82)

    if len(token) == 1:
        ch = token
        if ch == "=":
            return max(x_height * 0.72, 0.42)
        if ch.isupper():
            return max(cap_scale * 1.00, asc_desc_height * 1.12)
        if ch in TALL_BRACKETS:
            return max(cap_scale * 0.98, 0.90)
        if ch in ("f", "j"):
            return full_loop_height * (1.04 if ch == "j" else 1.00)
        if ch == "J":
            return max(cap_scale * 1.12, full_loop_height)
        if ch == "i":
            return asc_desc_height * 1.02
        if ch in ASCENDERS or ch in DESCENDERS or ch.isdigit() or ch == "?":
            return asc_desc_height
        if ch in SMALL_PUNCT:
            return max(x_height * 0.45, 0.24)
        return x_height

    has_cap = any(c.isupper() for c in token)
    has_asc = any(c in ASCENDERS or c in ("i", "f") for c in token)
    has_desc = any(c in DESCENDERS or c == "f" for c in token)

    if has_cap and has_desc:
        return max(full_loop_height, cap_scale * 1.24)
    if has_cap:
        return max(cap_scale * 1.02, asc_desc_height * 1.14)
    if has_asc and has_desc:
        return full_loop_height * 0.96
    if has_asc or has_desc:
        return asc_desc_height * 1.02
    return x_height


def _load_single_glyph_image(img_path: Path) -> Image.Image | None:
    try:
        im = Image.open(img_path).convert("RGBA")
        if im.width <= 1 or im.height <= 1:
            return None
        return im
    except Exception:
        return None


def load_raw_glyph_bank(glyphs_root: Path) -> dict[str, list[Image.Image]]:
    """Load cropped glyph images organized into subfolders under glyphs_root."""
    bank: dict[str, list[Image.Image]] = {}
    if not glyphs_root.exists() or not glyphs_root.is_dir():
        return bank

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for sub in sorted(glyphs_root.iterdir()):
        if not sub.is_dir():
            continue
        folder_name = sub.name
        imgs: list[Image.Image] = []
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in exts:
                loaded = _load_single_glyph_image(f)
                if loaded is not None:
                    imgs.append(loaded)
        if imgs:
            bank[folder_name] = imgs
    return bank


def build_zip_from_bank(bank: dict[str, list[Image.Image]]) -> bytes:
    """Pack glyph images from bank[folder_name] into an in-memory ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder_name, imgs in sorted(bank.items()):
            for idx, img in enumerate(imgs, start=1):
                img_buf = io.BytesIO()
                rgba = img.convert("RGBA") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGBA")
                rgba.save(img_buf, format="PNG")
                zf.writestr(f"{folder_name}/{folder_name}_{idx:03d}.png", img_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


def parse_colored_segments(
    line_text: str, default_rgb: tuple[int, int, int]
) -> list[tuple[str, tuple[int, int, int]]]:
    """Split text into (substring, rgb) using [red]...[/red], [blue]...[/blue], [green]...[/green], [black]...[/black]."""
    segments: list[tuple[str, tuple[int, int, int]]] = []
    pattern = re.compile(r"\[(red|blue|green|black)\](.*?)\[/\1\]", flags=re.IGNORECASE | re.DOTALL)
    pos = 0
    for m in pattern.finditer(line_text):
        if m.start() > pos:
            plain = line_text[pos : m.start()]
            if plain:
                segments.append((plain, default_rgb))
        tag_color = m.group(1).lower()
        inner = m.group(2)
        if inner:
            segments.append((inner, INK_PALETTE.get(tag_color, default_rgb)))
        pos = m.end()
    if pos < len(line_text):
        tail = line_text[pos:]
        if tail:
            segments.append((tail, default_rgb))
    if not segments:
        segments.append((line_text, default_rgb))
    return segments


def style_glyph_rgba(
    raw_im: Image.Image,
    ink_rgb: tuple[int, int, int] = (28, 30, 36),
    opacity: float = 1.0,
    stroke_delta: int = 0,
) -> Image.Image | None:
    """Convert a cropped glyph image to transparent RGBA tinted with ink_rgb, opacity, and stroke weight."""
    rgba = raw_im.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    if arr.size == 0:
        return None

    rgb_arr = arr[:, :, :3].astype(np.float32)
    alpha_ch = arr[:, :, 3]

    # Treat near-white pixels as background if image has opaque background, with strong ink contrast
    if np.median(alpha_ch) > 240:
        luminance = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        darkness = np.clip((244.0 - luminance) / 105.0, 0.0, 1.0) ** 0.70
        out_alpha = (darkness * 255.0).astype(np.uint8)
    else:
        norm_a = np.clip(alpha_ch.astype(np.float32) / 255.0, 0.0, 1.0) ** 0.75
        out_alpha = (norm_a * 255.0).astype(np.uint8)

    if stroke_delta != 0:
        k_size = max(3, abs(stroke_delta) * 2 + 1)
        kernel = np.ones((k_size, k_size), dtype=np.uint8)
        if stroke_delta > 0:
            out_alpha = cv2.dilate(out_alpha, kernel, iterations=1)
        else:
            out_alpha = cv2.erode(out_alpha, kernel, iterations=1)

    boosted_alpha = np.where(
        out_alpha > 18,
        np.clip((out_alpha.astype(np.float32) / 255.0) ** 0.25 * 255.0 * float(opacity), 0, 255),
        0.0,
    )
    final_alpha = boosted_alpha.astype(np.uint8)
    # Deepen near-black default ink so strokes read as rich black ink
    if ink_rgb[0] < 50 and ink_rgb[1] < 50 and ink_rgb[2] < 50:
        ink_rgb = (10, 11, 14)
    ys, xs = np.where(final_alpha > 18)
    if len(ys) == 0 or len(xs) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    out = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    out[:, :, 0] = ink_rgb[0]
    out[:, :, 1] = ink_rgb[1]
    out[:, :, 2] = ink_rgb[2]
    out[:, :, 3] = final_alpha[y0:y1, x0:x1]
    return Image.fromarray(out, mode="RGBA")


def make_background_page(
    baselines: list[int],
    bg_mode: str = "Plain White",
    custom_bg: Image.Image | None = None,
    margin_side: int = 115,
) -> Image.Image:
    """Create a PAGE_W x PAGE_H RGBA page with optional ruled lines or custom background."""
    if bg_mode == "Uploaded Image" and custom_bg is not None:
        base_rgb = custom_bg.convert("RGB").resize((PAGE_W, PAGE_H), Image.Resampling.BICUBIC)
    elif bg_mode == "Beige Ruled":
        base_rgb = Image.new("RGB", (PAGE_W, PAGE_H), (247, 241, 227))
    elif bg_mode == "Ruled White":
        base_rgb = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    else:
        base_rgb = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))

    if bg_mode in ("Ruled White", "Beige Ruled"):
        d = ImageDraw.Draw(base_rgb)
        rule_color = (210, 222, 238) if bg_mode == "Ruled White" else (218, 206, 184)
        margin_line_color = (232, 165, 165)
        for by in baselines:
            if MARGIN_TOP <= by <= PAGE_H - MARGIN_BOTTOM + 20:
                d.line([(50, by), (PAGE_W - 50, by)], fill=rule_color, width=1)
        mx = max(55, margin_side - 20)
        d.line([(mx, 40), (mx, PAGE_H - 40)], fill=margin_line_color, width=1)

    return base_rgb.convert("RGBA")


def draw_sheet_fiducial(draw: ImageDraw.ImageDraw, sheet_index: int, right_x: int, top_y: int = 24) -> None:
    """Draw a 6-bit binary fiducial marker in the top-right corner for sheet auto-detection."""
    box_size = 18
    gap = 6
    start_x = right_x - (6 * box_size + 5 * gap)
    for bit in range(6):
        is_one = ((sheet_index >> (5 - bit)) & 1) == 1
        x0 = start_x + bit * (box_size + gap)
        y0 = top_y
        x1 = x0 + box_size
        y1 = y0 + box_size
        if is_one:
            draw.rectangle([x0, y0, x1, y1], fill=(15, 23, 42), outline=(15, 23, 42), width=2)
        else:
            draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(15, 23, 42), width=2)


def _word_or_token_list(items: list[str], prefix: str = "word_") -> list[str]:
    out: list[str] = []
    for it in items:
        if it.startswith(("flow_", "phys_", "math_", "lig_")):
            out.append(it)
        else:
            out.append(f"{prefix}{it.replace(' ', '_')}")
    return out


def get_sheet_labels(sheet_index: int) -> tuple[str, list[str]]:
    title, _ = SHEET_INFO[sheet_index]
    mapping: dict[int, list[str]] = {
        1: SHEET_1_LABELS,
        2: [f"lig_{w}" for w in SHEET_2_DOUBLE_PAIRS],
        3: [f"lig_{w}" for w in SHEET_3_TRIPLE_WORDS],
        4: [f"lig_{w}" for w in SHEET_4_COMMON_4],
        5: [f"lig_{w}" for w in SHEET_5_COMMON_5],
        6: SHEET_6_MATH_GREEK,
        7: SHEET_7_MATH_OPERATORS,
        8: SHEET_8_MATH_BRACKETS_NOTATION,
        9: SHEET_9_MATH_COMMON_TERMS,
        10: SHEET_10_FLOW_NODES,
        11: SHEET_11_FLOW_ARROWS,
        12: _word_or_token_list(SHEET_12_ENG_6LETTER),
        13: _word_or_token_list(SHEET_13_ENG_7LETTER),
        14: _word_or_token_list(SHEET_14_ENG_AFFIXES, prefix="affix_"),
        15: _word_or_token_list(SHEET_15_ENG_VERBS_ADJ),
        16: _word_or_token_list(SHEET_16_ENG_CONNECTORS),
        17: _word_or_token_list(SHEET_17_ENG_PHRASES, prefix="phrase_"),
        18: SHEET_18_ENG_PUNCT_MISC,
        19: _word_or_token_list(SHEET_19_CS_DBMS_SQL, prefix="cs_db_"),
        20: _word_or_token_list(SHEET_20_CS_NETWORKS, prefix="cs_net_"),
        21: _word_or_token_list(SHEET_21_CS_OS_DSA, prefix="cs_os_"),
        22: _word_or_token_list(SHEET_22_ENG_GENERAL_TECH, prefix="cs_tech_"),
        23: SHEET_23_FLOW_DECISIONS,
        24: SHEET_24_FLOW_ARCH_NODES,
        25: SHEET_25_FLOW_CONNECTORS_EX,
        26: _word_or_token_list(SHEET_26_ENG_8LETTER),
        27: _word_or_token_list(SHEET_27_ENG_9PLUS_LETTER),
        28: _word_or_token_list(SHEET_28_ENG_COMMON_PAIRS_2, prefix="lig_"),
        29: _word_or_token_list(SHEET_29_ENG_COMMON_WORDS_6),
        30: _word_or_token_list(SHEET_30_ENG_COMMON_WORDS_7),
        31: _word_or_token_list(SHEET_31_ENG_ACADEMIC_NOTES),
        32: SHEET_32_MATH_LINEAR_ALG,
        33: SHEET_33_MATH_CALC_STATS,
        34: SHEET_34_MATH_DISCRETE_LOGIC,
        35: _word_or_token_list(SHEET_35_CS_ALGO_DS, prefix="cs_algo_"),
        36: _word_or_token_list(SHEET_36_CS_SYSTEMS_ARCH, prefix="cs_sys_"),
        37: _word_or_token_list(SHEET_37_CS_AI_ML, prefix="cs_ai_"),
        38: _word_or_token_list(SHEET_38_ENG_CIRCUITS_SIGNALS, prefix="eng_ee_"),
        39: _word_or_token_list(SHEET_39_PHYSICS_MECHANICS, prefix="phys_"),
        40: _word_or_token_list(SHEET_40_PHYSICS_EM_OPTICS, prefix="phys_"),
        41: SHEET_41_PHYSICS_SYMBOLS,
        42: _word_or_token_list(SHEET_42_BIO_CELL_GENETICS, prefix="bio_"),
        43: _word_or_token_list(SHEET_43_BIO_PHYSIOLOGY, prefix="bio_"),
        44: _word_or_token_list(SHEET_44_ENG_CODE_TOKENS, prefix="code_"),
        45: SHEET_45_ENG_FLOW_EXTRAS,
    }
    return title, mapping[sheet_index]


def build_template_sheet(sheet_index: int) -> Image.Image:
    """Render a printable A4-proportioned template grid sheet for sheet_index (1..45)."""
    sheet_w, sheet_h = 2480, 1754
    img = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_str, desc_str = SHEET_INFO.get(sheet_index, (f"Sheet {sheet_index:02d}", ""))
    header_font = load_crisp_font(32)
    sub_font = load_crisp_font(20)
    cell_font = load_crisp_font(17)

    draw.text((60, 22), f"{title_str}  [SHEET {sheet_index}]", fill=(15, 23, 42), font=header_font)
    draw.text((60, 64), desc_str, fill=(71, 85, 105), font=sub_font)
    draw_sheet_fiducial(draw, sheet_index=sheet_index, right_x=sheet_w - 60, top_y=24)

    _, labels = get_sheet_labels(sheet_index)
    cols = 10
    grid_left = 60
    grid_right = sheet_w - 60
    grid_top = 108
    grid_bottom = sheet_h - 60
    cell_w = (grid_right - grid_left) // cols
    rows = max(1, (len(labels) + cols - 1) // cols)
    cell_h = max(120, min(320, (grid_bottom - grid_top) // rows))

    for idx, label in enumerate(labels):
        r = idx // cols
        c = idx % cols
        x0 = grid_left + c * cell_w
        y0 = grid_top + r * cell_h
        x1 = x0 + cell_w
        y1 = y0 + cell_h
        if y1 > grid_bottom:
            break

        draw.rectangle([x0, y0, x1, y1], outline=(71, 85, 105), width=2)
        badge_h = 30
        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y0 + badge_h], fill=(226, 232, 240))
        draw.text((x0 + 6, y0 + 6), display_label(label), fill=(15, 23, 42), font=cell_font)

    return img


def build_template_pack_zip(sheet_indices: list[int], subfolder_name: str = "templates") -> bytes:
    """Create a ZIP containing PNG template sheets for the requested sheet numbers."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for s_num in sheet_indices:
            im = build_template_sheet(int(s_num))
            p_buf = io.BytesIO()
            im.save(p_buf, format="PNG")
            zf.writestr(f"{subfolder_name}/template_sheet_{int(s_num):02d}.png", p_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


def detect_sheet_index(pil_img: Image.Image, filename: str = "") -> int:
    """Detect sheet index (1..45) from top-right 6-bit fiducial marker or filename fallback."""
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    arr = np.array(rgb)

    # Sample top-right fiducial strip
    y_band = arr[int(h * 0.01) : int(h * 0.05), int(w * 0.80) : int(w * 0.99)]
    if y_band.size > 0:
        gray = cv2.cvtColor(y_band, cv2.COLOR_RGB2GRAY)
        # Look for dark squares along top-right
        h_proj = (gray < 110).mean(axis=0)
        active_cols = np.where(h_proj > 0.25)[0]
        if len(active_cols) >= 6:
            # Split active span into 6 equal bins
            span_start, span_end = int(active_cols[0]), int(active_cols[-1])
            bin_w = max(1.0, (span_end - span_start + 1) / 6.0)
            detected = 0
            for b in range(6):
                c0 = int(span_start + b * bin_w)
                c1 = max(c0 + 1, int(span_start + (b + 1) * bin_w))
                fill_ratio = float((gray[:, c0:c1] < 110).mean())
                if fill_ratio > 0.45:
                    detected |= (1 << (5 - b))
            if 1 <= detected <= 45:
                return detected

    m = re.search(r"(?:sheet[_\s-]*)?0?([1-9]|[1-3][0-9]|4[0-5])", filename.lower())
    if m:
        val = int(m.group(1))
        if 1 <= val <= 45:
            return val
    return 1


def extract_from_template_image(
    pil_img: Image.Image, sheet_index: int = 1
) -> dict[str, list[Image.Image]]:
    """Extract handwritten glyph crops from each grid cell of a filled template sheet."""
    _, labels = get_sheet_labels(int(sheet_index))
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Detect grid bounding box via dark border lines
    row_dark = (gray < 140).mean(axis=1)
    col_dark = (gray < 140).mean(axis=0)
    r_lines = np.where(row_dark > 0.35)[0]
    c_lines = np.where(col_dark > 0.25)[0]

    grid_top = int(r_lines[0]) if len(r_lines) >= 2 else int(h * 0.06)
    grid_bottom = int(r_lines[-1]) if len(r_lines) >= 2 else int(h * 0.96)
    grid_left = int(c_lines[0]) if len(c_lines) >= 2 else int(w * 0.025)
    grid_right = int(c_lines[-1]) if len(c_lines) >= 2 else int(w * 0.975)

    cols = 10
    rows = max(1, (len(labels) + cols - 1) // cols)
    cell_w = (grid_right - grid_left) / cols
    cell_h = (grid_bottom - grid_top) / rows

    extracted: dict[str, list[Image.Image]] = {}
    for idx, label in enumerate(labels):
        r = idx // cols
        c = idx % cols
        x0 = int(round(grid_left + c * cell_w)) + 8
        x1 = int(round(grid_left + (c + 1) * cell_w)) - 8
        y0 = int(round(grid_top + r * cell_h)) + 34  # skip header badge
        y1 = int(round(grid_top + (r + 1) * cell_h)) - 8
        if x1 <= x0 + 8 or y1 <= y0 + 8 or y0 >= h or x0 >= w:
            continue

        cell_crop = arr[y0 : min(h, y1), x0 : min(w, x1)]
        cell_gray = cv2.cvtColor(cell_crop, cv2.COLOR_RGB2GRAY)
        ink_mask = cell_gray < 155
        ink_mask = cv2.erode(ink_mask.astype(np.uint8), np.ones((2, 2), np.uint8), iterations=0) > 0

        if int(ink_mask.sum()) < 25:
            continue

        ys, xs = np.where(ink_mask)
        by0, by1 = max(0, int(ys.min()) - 3), min(cell_crop.shape[0], int(ys.max()) + 4)
        bx0, bx1 = max(0, int(xs.min()) - 3), min(cell_crop.shape[1], int(xs.max()) + 4)
        tight = cell_crop[by0:by1, bx0:bx1]
        if tight.shape[0] < 6 or tight.shape[1] < 6:
            continue

        rgba_crop = Image.fromarray(tight, mode="RGB").convert("RGBA")
        styled = style_glyph_rgba(rgba_crop, ink_rgb=(28, 30, 36), opacity=1.0, stroke_delta=0)
        if styled is not None:
            extracted.setdefault(label, []).append(styled)

    return extracted


def save_bank_to_folder(bank: dict[str, list[Image.Image]], glyphs_root: Path) -> int:
    """Save all glyph images in bank to glyphs_root/<folder_name>/<folder_name>_###.png."""
    glyphs_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder_name, imgs in sorted(bank.items()):
        sub = glyphs_root / folder_name
        sub.mkdir(parents=True, exist_ok=True)
        for idx, im in enumerate(imgs, start=1):
            out_path = sub / f"{folder_name}_{idx:03d}.png"
            im.convert("RGBA").save(out_path, format="PNG")
            count += 1
    return count


def load_bank_from_zip(zip_bytes: bytes) -> dict[str, list[Image.Image]]:
    """Load glyph PNG/JPG images grouped by top-level subfolder from a ZIP archive."""
    loaded_bank: dict[str, list[Image.Image]] = {}
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for entry in sorted(zf.namelist()):
            if entry.endswith("/") or entry.startswith("__MACOSX"):
                continue
            p = Path(entry)
            if p.suffix.lower() not in exts or len(p.parts) < 2:
                continue
            folder_name = p.parts[-2]
            raw_im = Image.open(io.BytesIO(zf.read(entry))).convert("RGBA")
            if raw_im.width > 1 and raw_im.height > 1:
                loaded_bank.setdefault(folder_name, []).append(raw_im)
    return loaded_bank


def parse_script_chunks(expr: str) -> list[tuple[str, str]]:
    """Parse ^ and _ scripts with brace-depth tracking (e.g. y^2, z^{n+1}, x_{i+1}, a^2)."""
    chunks: list[tuple[str, str]] = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] in ("^", "_") and i + 1 < n:
            mode = "sup" if expr[i] == "^" else "sub"
            if expr[i + 1] == "{":
                depth = 1
                j = i + 2
                while j < n and depth > 0:
                    if expr[j] == "{":
                        depth += 1
                    elif expr[j] == "}":
                        depth -= 1
                    j += 1
                inner = expr[i + 2 : j - 1] if depth == 0 else expr[i + 2 : j]
                chunks.append((inner, mode))
                i = j
            else:
                chunks.append((expr[i + 1], mode))
                i += 2
        else:
            chunks.append((expr[i], "normal"))
            i += 1
    return chunks


def render_handwritten_pages(
    text: str,
    raw_bank: dict[str, list[Image.Image]],
    seed: int = 42,
    base_size: int = 33,
    cap_scale: float = 1.00,
    lower_scale: float = 0.48,
    width_scale: float = 1.00,
    line_stretch: float = 1.00,
    script_scale: float = 0.72,
    script_height_offset: float = 0.25,
    char_scale_map: dict[str, float] | None = None,
    char_spacing: int = -2,
    word_spacing: int = 19,
    line_gap: int = 64,
    margin_side: int = 115,
    baseline_jitter: int = 1,
    tilt_deg: float = 0.8,
    opacity: float = 1.00,
    stroke_delta: int = 0,
    default_rgb: tuple[int, int, int] = (28, 30, 36),
    bg_mode: str = "Plain White",
    custom_bg: Image.Image | None = None,
) -> list[Image.Image]:
    rng = random.Random(seed)
    # Prevent adjacent lines from colliding when base_size is increased
    effective_line_gap = max(line_gap, int(round(base_size * 1.65)))
    usable_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
    lines_per_page = max(1, usable_h // effective_line_gap)
    page_baselines = [MARGIN_TOP + (i + 1) * line_gap for i in range(lines_per_page)]

    char_overrides = char_scale_map or {}
    styled_cache: dict[tuple[str, tuple[int, int, int]], list[Image.Image]] = {}

    def get_styled_variants(folder: str, rgb: tuple[int, int, int]) -> list[Image.Image]:
        key = (folder, rgb)
        if key not in styled_cache:
            out: list[Image.Image] = []
            for raw_im in raw_bank.get(folder, []):
                g = style_glyph_rgba(raw_im, ink_rgb=rgb, opacity=opacity, stroke_delta=stroke_delta)
                if g is not None:
                    out.append(g)
            styled_cache[key] = out
        return styled_cache[key]

    def build_handwritten_snippet(snippet: str, rgb: tuple[int, int, int], scale_factor: float = 0.65) -> Image.Image:
        """Render a short expression or flowchart label using the user's handwritten glyphs."""
        glyph_items: list[tuple[str, Image.Image | None]] = []
        total_w = 0.0
        max_h = max(18, int(round(base_size * scale_factor)))
        eff_spacing = max(-4, int(round(char_spacing * scale_factor)))

        for chunk_str, script_mode in parse_script_chunks(snippet):
            s_mult = 0.65 if script_mode in ("sup", "sub") else 1.0
            for tok, folder in tokenize_word_with_ligatures(chunk_str, raw_bank):
                variants = get_styled_variants(folder, rgb) if folder else []
                if variants:
                    raw_g = rng.choice(variants)
                    mult = token_height_multiplier(tok, lower_scale=lower_scale, cap_scale=cap_scale) * scale_factor * s_mult
                    th = max(8, int(round(base_size * mult)))
                    sc = th / max(raw_g.height, 1)
                    nw = max(4, int(round(raw_g.width * sc)))
                    nh = max(4, int(round(raw_g.height * sc)))
                    g_im = raw_g.resize((nw, nh), Image.Resampling.BICUBIC)
                    glyph_items.append((tok, g_im))
                    total_w += nw + eff_spacing
                    max_h = max(max_h, nh + int(base_size * 0.25 * scale_factor))
                else:
                    fw = max(6.0, base_size * 0.45 * scale_factor * len(tok))
                    glyph_items.append((tok, None))
                    total_w += fw + eff_spacing

        canvas_w = max(16, int(round(total_w)) + 8)
        canvas_h = max(22, int(round(max_h)) + 10)
        out_img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        d_out = ImageDraw.Draw(out_img)
        fallback_f = load_crisp_font(max(12, int(base_size * 0.42 * scale_factor)))
        cx = 2.0
        base_line = canvas_h - 6.0

        for tok, g_im in glyph_items:
            if g_im is not None:
                has_desc = any(c in DESCENDERS or c == "f" for c in tok) and not tok.isupper()
                py = int(round(base_line - g_im.height * (0.65 if has_desc else 1.0)))
                out_img.paste(g_im, (int(round(cx)), max(0, py)), g_im)
                cx += g_im.width + eff_spacing
            else:
                d_out.text((int(round(cx)), int(round(base_line - base_size * 0.42 * scale_factor))), tok, fill=(*rgb, int(255 * opacity)), font=fallback_f)
                cx += max(6.0, base_size * 0.45 * scale_factor * len(tok)) + eff_spacing

        bbox = out_img.getbbox()
        return out_img.crop(bbox) if bbox else out_img

    pages: list[Image.Image] = [make_background_page(page_baselines, bg_mode, custom_bg, margin_side)]
    line_idx = 0
    cursor_x = float(margin_side)
    right_limit = PAGE_W - margin_side

    def ensure_next_line() -> None:
        nonlocal line_idx, cursor_x
        cursor_x = float(margin_side)
        line_idx += 1
        if line_idx >= len(page_baselines):
            pages.append(make_background_page(page_baselines, bg_mode, custom_bg, margin_side))
            line_idx = 0

    def draw_sketch_box(page_img: Image.Image, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
        d = ImageDraw.Draw(page_img)
        for pass_i in range(2):
            pts = [
                (x0 + rng.randint(-2, 2), y0 + rng.randint(-2, 2)),
                (x1 + rng.randint(-2, 2), y0 + rng.randint(-2, 2)),
                (x1 + rng.randint(-2, 2), y1 + rng.randint(-2, 2)),
                (x0 + rng.randint(-2, 2), y1 + rng.randint(-2, 2)),
                (x0 + rng.randint(-2, 2), y0 + rng.randint(-2, 2)),
            ]
            d.line(pts, fill=rgb, width=2 if pass_i == 0 else 1)

    def draw_flow_node(
        page_img: Image.Image,
        step_text: str,
        x0: int,
        y0: int,
        w: int,
        h: int,
        rgb: tuple[int, int, int],
    ) -> None:
        d = ImageDraw.Draw(page_img)
        t_low = step_text.strip().lower()
        x1, y1 = x0 + w, y0 + h
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

        if t_low.startswith("?") or t_low.startswith("if ") or t_low.endswith("?"):
            drawn = get_styled_variants("flow_diamond", rgb)
            if drawn:
                g = rng.choice(drawn).resize((w, h), Image.Resampling.BICUBIC)
                page_img.paste(g, (x0, y0), g)
            else:
                pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy), (cx, y0)]
                d.line(pts, fill=rgb, width=2)
        elif t_low in ("start", "end", "stop") or t_low.startswith("start ") or t_low.startswith("end "):
            drawn = get_styled_variants("flow_oval", rgb)
            if drawn:
                g = rng.choice(drawn).resize((w, h), Image.Resampling.BICUBIC)
                page_img.paste(g, (x0, y0), g)
            else:
                d.ellipse([x0, y0, x1, y1], outline=rgb, width=2)
        elif t_low.startswith("io:") or t_low.startswith("input ") or t_low.startswith("output "):
            drawn = get_styled_variants("flow_parallelogram", rgb)
            if drawn:
                g = rng.choice(drawn).resize((w, h), Image.Resampling.BICUBIC)
                page_img.paste(g, (x0, y0), g)
            else:
                shift = max(10, w // 7)
                pts = [(x0 + shift, y0), (x1, y0), (x1 - shift, y1), (x0, y1), (x0 + shift, y0)]
                d.line(pts, fill=rgb, width=2)
        elif t_low.startswith("db:") or "database" in t_low:
            drawn = get_styled_variants("flow_cylinder", rgb)
            if drawn:
                g = rng.choice(drawn).resize((w, h), Image.Resampling.BICUBIC)
                page_img.paste(g, (x0, y0), g)
            else:
                ry = max(6, h // 6)
                d.ellipse([x0, y0, x1, y0 + ry * 2], outline=rgb, width=2)
                d.line([(x0, y0 + ry), (x0, y1 - ry)], fill=rgb, width=2)
                d.line([(x1, y0 + ry), (x1, y1 - ry)], fill=rgb, width=2)
                d.arc([x0, y1 - ry * 2, x1, y1], start=0, end=180, fill=rgb, width=2)
        else:
            drawn = get_styled_variants("flow_rect", rgb)
            if drawn:
                g = rng.choice(drawn).resize((w, h), Image.Resampling.BICUBIC)
                page_img.paste(g, (x0, y0), g)
            else:
                draw_sketch_box(page_img, x0, y0, x1, y1, rgb)

    def render_flowchart_line(flow_spec: str, rgb: tuple[int, int, int]) -> None:
        nonlocal line_idx, cursor_x
        steps = [s.strip() for s in re.split(r"\s*->\s*|\s*=>\s*", flow_spec) if s.strip()]
        if not steps:
            return
        if cursor_x > float(margin_side) + 2.0:
            ensure_next_line()
        if line_idx + 1 >= len(page_baselines):
            ensure_next_line()

        def _row_bounds(b_idx: int) -> tuple[int, int, int]:
            t_y = int(page_baselines[b_idx] - base_size * 1.05)
            b_y = int(page_baselines[b_idx] + base_size * 0.65)
            return t_y, b_y, max(56, b_y - t_y)

        box_top, box_bottom, box_h = _row_bounds(line_idx)
        cur_x = int(margin_side)
        arrow_len = 34
        drawn_arrows = get_styled_variants("flow_arr_right", rgb)
        pending_wrap_from: tuple[int, int] | None = None

        for s_i, raw_step in enumerate(steps):
            step_segs = parse_colored_segments(raw_step, rgb)
            rendered_pieces: list[Image.Image] = []
            pieces_w = 0
            for seg_t, seg_c in step_segs:
                piece = build_handwritten_snippet(seg_t, seg_c, scale_factor=0.86)
                rendered_pieces.append(piece)
                pieces_w += piece.width + 3

            clean_step_label = "".join(t for t, _ in step_segs)
            t_check = clean_step_label.strip().lower()
            is_diamond = t_check.startswith(("?", "if ")) or t_check.endswith("?")
            is_para = t_check.startswith(("io:", "input ", "output "))
            pad_x = 84 if is_diamond else (68 if is_para else 54)
            est_box_w = max(124, int(pieces_w) + pad_x)
            arrow_gap = 12 if s_i < len(steps) - 1 else 0
            needed_w = est_box_w + arrow_gap + (arrow_len if s_i < len(steps) - 1 else 0)

            if cur_x + needed_w > PAGE_W - margin_side and s_i > 0:
                prev_mid_y = (box_top + box_bottom) // 2
                prev_right_x = cur_x
                ensure_next_line()
                ensure_next_line()
                box_top, box_bottom, box_h = _row_bounds(line_idx)
                cur_x = int(margin_side)
                pending_wrap_from = (prev_right_x, prev_mid_y)

            if pending_wrap_from is not None:
                p_x, p_y = pending_wrap_from
                new_mid_y = (box_top + box_bottom) // 2
                mid_drop_y = (p_y + new_mid_y) // 2
                right_turn_x = p_x + 18
                left_stub_x = max(24, cur_x - 26)

                def _draw_hand_seg(pts_list: list[tuple[int, int]]) -> None:
                    d_conn = ImageDraw.Draw(pages[-1])
                    for pass_i in range(2):
                        jittered = [
                            (px + rng.randint(-1, 1), py + rng.randint(-1, 1))
                            for px, py in pts_list
                        ]
                        d_conn.line(jittered, fill=rgb, width=2 if pass_i == 0 else 1)

                # Right stub from previous row
                h_glyphs = get_styled_variants("flow_line_h", rgb)
                v_glyphs = get_styled_variants("flow_line_v", rgb)
                edr_glyphs = get_styled_variants("flow_elbow_dr", rgb)
                erd_glyphs = get_styled_variants("flow_elbow_rd", rgb)

                if h_glyphs and v_glyphs and edr_glyphs and erd_glyphs and drawn_arrows:
                    # 1. Right turn elbow
                    g_edr = rng.choice(edr_glyphs).resize((20, 20), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_edr, (p_x, p_y - 10), g_edr)
                    # 2. Drop down to mid_drop_y
                    drop_h1 = max(8, mid_drop_y - (p_y + 10))
                    g_v1 = rng.choice(v_glyphs).resize((10, drop_h1), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_v1, (right_turn_x - 5, p_y + 10), g_v1)
                    # 3. Horizontal return across page
                    span_w = max(20, right_turn_x - left_stub_x)
                    g_h = rng.choice(h_glyphs).resize((span_w, 10), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_h, (left_stub_x, mid_drop_y - 5), g_h)
                    # 4. Drop down to new row mid_y
                    drop_h2 = max(8, new_mid_y - mid_drop_y - 10)
                    g_v2 = rng.choice(v_glyphs).resize((10, drop_h2), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_v2, (left_stub_x - 5, mid_drop_y), g_v2)
                    # 5. Elbow into arrow pointing at Process
                    g_erd = rng.choice(erd_glyphs).resize((16, 16), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_erd, (left_stub_x - 4, new_mid_y - 10), g_erd)
                    g_arr = rng.choice(drawn_arrows).resize((22, 18), Image.Resampling.BICUBIC)
                    pages[-1].paste(g_arr, (cur_x - 22, new_mid_y - 9), g_arr)
                else:
                    _draw_hand_seg([
                        (p_x, p_y),
                        (right_turn_x, p_y),
                        (right_turn_x, mid_drop_y),
                        (left_stub_x, mid_drop_y),
                        (left_stub_x, new_mid_y),
                        (cur_x, new_mid_y),
                    ])
                    _draw_hand_seg([
                        (cur_x - 8, new_mid_y - 5),
                        (cur_x, new_mid_y),
                        (cur_x - 8, new_mid_y + 5),
                    ])
                pending_wrap_from = None

            draw_flow_node(pages[-1], clean_step_label, cur_x, box_top, est_box_w, box_h, rgb)

            text_x = cur_x + max(18, (est_box_w - pieces_w) // 2)
            for piece in rendered_pieces:
                py_c = box_top + max(6, (box_h - piece.height) // 2)
                pages[-1].paste(piece, (text_x, py_c), piece)
                text_x += piece.width + 3
            cur_x += est_box_w

            if s_i < len(steps) - 1:
                if cur_x + arrow_gap + arrow_len + 80 <= PAGE_W - margin_side:
                    cur_x += arrow_gap
                    mid_y = (box_top + box_bottom) // 2
                    if drawn_arrows:
                        arr_glyph = rng.choice(drawn_arrows).resize((arrow_len, 22), Image.Resampling.BICUBIC)
                        pages[-1].paste(arr_glyph, (cur_x, mid_y - 11), arr_glyph)
                    else:
                        d_arr = ImageDraw.Draw(pages[-1])
                        d_arr.line([(cur_x, mid_y), (cur_x + arrow_len, mid_y)], fill=rgb, width=2)
                        d_arr.line(
                            [(cur_x + arrow_len - 7, mid_y - 5), (cur_x + arrow_len, mid_y), (cur_x + arrow_len - 7, mid_y + 5)],
                            fill=rgb,
                            width=2,
                        )
                    cur_x += arrow_len
        ensure_next_line()

    def render_stacked_fraction_token(
        num_str: str,
        den_str: str,
        seg_rgb: tuple[int, int, int],
        line_slope: float,
    ) -> None:
        nonlocal cursor_x
        num_img = build_handwritten_snippet(num_str, seg_rgb, scale_factor=0.96)
        den_img = build_handwritten_snippet(den_str, seg_rgb, scale_factor=0.96)
        frac_w = max(40, num_img.width + 14, den_img.width + 14)

        if cursor_x + frac_w > right_limit and cursor_x > float(margin_side) + 2.0:
            ensure_next_line()

        base_y = float(page_baselines[line_idx]) + (cursor_x - margin_side) * line_slope
        bar_y = int(round(base_y - base_size * 0.34))
        x0 = int(round(cursor_x))
        d = ImageDraw.Draw(pages[-1])

        num_x = x0 + max(2, (frac_w - num_img.width) // 2)
        num_y = max(4, bar_y - num_img.height - 5)
        pages[-1].paste(num_img, (num_x, num_y), num_img)

        d.line([(x0 + 2, bar_y), (x0 + frac_w - 2, bar_y)], fill=seg_rgb, width=2)

        den_x = x0 + max(2, (frac_w - den_img.width) // 2)
        den_y = bar_y + 5
        pages[-1].paste(den_img, (den_x, den_y), den_img)
        cursor_x += frac_w + max(8, word_spacing // 2)

    def render_radical_token(inner_str: str, seg_rgb: tuple[int, int, int], line_slope: float) -> None:
        nonlocal cursor_x
        inner_img = build_handwritten_snippet(inner_str, seg_rgb, scale_factor=0.95)
        total_w = max(48, inner_img.width + 28)

        if cursor_x + total_w > right_limit and cursor_x > float(margin_side) + 2.0:
            ensure_next_line()

        base_y = float(page_baselines[line_idx]) + (cursor_x - margin_side) * line_slope
        top_y = int(round(base_y - base_size * 0.92))
        bot_y = int(round(base_y - base_size * 0.02))
        x0 = int(round(cursor_x))
        d = ImageDraw.Draw(pages[-1])

        sqrt_variants = get_styled_variants("math_sqrt", seg_rgb)
        if sqrt_variants:
            rad_h = max(24, bot_y - top_y)
            rad_w = max(18, int(base_size * 0.45))
            rg = rng.choice(sqrt_variants).resize((rad_w, rad_h), Image.Resampling.BICUBIC)
            pages[-1].paste(rg, (x0, top_y), rg)
            bar_start_x = x0 + rad_w - 2
            total_w = max(48, (bar_start_x - x0) + inner_img.width + 10)
            for _ in range(2):
                jy = rng.randint(-1, 1)
                d.line([(bar_start_x, top_y + jy), (x0 + total_w, top_y + jy)], fill=seg_rgb, width=2)
            pages[-1].paste(inner_img, (bar_start_x + 4, top_y + 5), inner_img)
        else:
            for _ in range(2):
                jx = rng.randint(-1, 1)
                jy = rng.randint(-1, 1)
                d.line(
                    [
                        (x0 + jx, (top_y + bot_y) // 2 + jy),
                        (x0 + 7 + jx, bot_y + jy),
                        (x0 + 15 + jx, top_y + jy),
                        (x0 + total_w - 2 + jx, top_y + jy),
                    ],
                    fill=seg_rgb,
                    width=2,
                )
            pages[-1].paste(inner_img, (x0 + 18, top_y + 5), inner_img)

        cursor_x += total_w + max(8, word_spacing // 2)

        
    line_hitboxes: list[dict[str, int]] = []
    active_src_line: int | None = None
    active_start_page: int = 0
    active_start_y: int = 0

    def _begin_src_line(src_i: int) -> None:
        nonlocal active_src_line, active_start_page, active_start_y
        _flush_src_line()
        active_src_line = src_i
        active_start_page = len(pages) - 1
        by = page_baselines[min(line_idx, len(page_baselines) - 1)]
        active_start_y = max(0, int(by - base_size * 1.35))

    def _flush_src_line() -> None:
        nonlocal active_src_line
        if active_src_line is None:
            return
        end_page = len(pages) - 1
        by_end = page_baselines[min(line_idx, len(page_baselines) - 1)]
        y_end = min(PAGE_H, int(by_end + max(18, base_size * 0.45)))
        if start_p := active_start_page:
            pass
        if active_start_page == end_page:
            line_hitboxes.append({
                "page": active_start_page + 1,
                "src_line": active_src_line,
                "y0": max(0, active_start_y),
                "y1": max(active_start_y + 24, y_end),
            })
        else:
            line_hitboxes.append({
                "page": active_start_page + 1,
                "src_line": active_src_line,
                "y0": max(0, active_start_y),
                "y1": PAGE_H - MARGIN_BOTTOM,
            })
            line_hitboxes.append({
                "page": end_page + 1,
                "src_line": active_src_line,
                "y0": MARGIN_TOP,
                "y1": max(MARGIN_TOP + 24, y_end),
            })
        active_src_line = None
    lines = text.splitlines() or [""]
    fallback_math_font = load_crisp_font(max(22, int(base_size * 0.88)))

    for l_idx, raw_line in enumerate(lines):
        if l_idx > 0:
            ensure_next_line()
        _begin_src_line(l_idx)

        stripped_line = raw_line.strip()
        flow_m = re.fullmatch(r"\[flow\](.*?)\[/flow\]", stripped_line, flags=re.IGNORECASE)
        if flow_m:
            render_flowchart_line(flow_m.group(1), default_rgb)
            continue

        is_boxed_line = False
        box_max_x = float(margin_side)
        box_start_line = line_idx
        box_m = re.fullmatch(r"\[box\](.*?)\[/box\]", stripped_line, flags=re.IGNORECASE)
        if box_m:
            raw_line = box_m.group(1)
            is_boxed_line = True
            box_start_line = line_idx

        def _replace_math_tag(m: re.Match) -> str:
            m_tok = m.group(1)
            return MATH_FALLBACK_UNICODE.get(m_tok, m_tok)

        raw_line = re.sub(r"<(math_[A-Za-z]+)>", _replace_math_tag, raw_line)

        line_slope = rng.uniform(-0.012, 0.012)
        segments = parse_colored_segments(raw_line, default_rgb)

        for seg_text, seg_rgb in segments:
            words = seg_text.split(" ")
            for w_idx, word in enumerate(words):
                if w_idx > 0:
                    eff_word_spacing = max(word_spacing, int(round(base_size * 0.45)))
                    cursor_x += eff_word_spacing + rng.uniform(-1.5, 2.0)
                    if cursor_x >= right_limit:
                        ensure_next_line()
                        line_slope = rng.uniform(-0.012, 0.012)

                if not word:
                    continue

                if word.startswith(r"\frac{"):
                    num_s, after_num = extract_braced_arg(word, len(r"\frac"))
                    if after_num < len(word) and word[after_num] == "{":
                        den_s, _ = extract_braced_arg(word, after_num)
                        render_stacked_fraction_token(num_s, den_s, seg_rgb, line_slope)
                        continue
                elif word.startswith(r"\sqrt{"):
                    inner_s, _ = extract_braced_arg(word, len(r"\sqrt"))
                    render_radical_token(inner_s, seg_rgb, line_slope)
                    continue

                script_chunks = parse_script_chunks(word)
                prepared: list[tuple[str, Image.Image | None, str]] = []
                est_w = 0.0

                for chunk_str, script_mode in script_chunks:
                    chunk_tokens = tokenize_word_with_ligatures(chunk_str, raw_bank)
                    is_script = script_mode in ("sup", "sub")
                    for tok, folder in chunk_tokens:
                        variants = get_styled_variants(folder, seg_rgb) if folder else []
                        if variants:
                            raw_g = rng.choice(variants)
                            base_mult = token_height_multiplier(tok, lower_scale=lower_scale, cap_scale=cap_scale)
                            char_mult = float(char_overrides.get(folder or tok, 1.0))
                            if is_script:
                                mult = max(base_mult * float(script_scale), max(lower_scale * 1.05, 0.42)) * char_mult
                            else:
                                mult = base_mult * char_mult
                            target_h = max(6, int(round(base_size * mult)))
                            scale = target_h / max(raw_g.height, 1)
                            nw = max(4, int(round(raw_g.width * scale * float(width_scale) * float(line_stretch))))
                            nh = max(4, int(round(raw_g.height * scale)))
                            g_img = raw_g.resize((nw, nh), Image.Resampling.BICUBIC)

                            if tilt_deg > 0 and tok not in SMALL_PUNCT:
                                angle = rng.uniform(-tilt_deg, tilt_deg)
                                g_img = g_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
                                bbox = g_img.getbbox()
                                if bbox:
                                    g_img = g_img.crop(bbox)

                            prepared.append((tok, g_img, script_mode))
                            est_w += max(4.0, g_img.width + char_spacing)
                        elif tok in UNICODE_TO_MATH_FOLDER:
                            sym_img = Image.new("RGBA", (int(base_size * 0.9), int(base_size * 0.9)), (0, 0, 0, 0))
                            sd = ImageDraw.Draw(sym_img)
                            sd.text((2, 2), tok, fill=(*seg_rgb, int(255 * opacity)), font=fallback_math_font)
                            sb = sym_img.getbbox()
                            if sb:
                                sym_img = sym_img.crop(sb)
                            prepared.append((tok, sym_img, script_mode))
                            est_w += max(6.0, sym_img.width + char_spacing)
                        else:
                            prepared.append((tok, None, script_mode))
                            est_w += max(6.0, base_size * lower_scale * 0.8 * len(tok) + char_spacing)

                if cursor_x + est_w > right_limit and cursor_x > float(margin_side) + 2.0:
                    if is_boxed_line:
                        box_max_x = max(box_max_x, cursor_x)
                    ensure_next_line()
                    line_slope = rng.uniform(-0.012, 0.012)

                base_y = float(page_baselines[line_idx])
                word_drift = rng.uniform(-1.0, 1.0) if baseline_jitter > 0 else 0.0

                for tok, g_img, script_mode in prepared:
                    if g_img is not None:
                        slope_y = (cursor_x - margin_side) * line_slope
                        jitter_y = word_drift + (
                            rng.uniform(-baseline_jitter * 0.5, baseline_jitter * 0.5) if baseline_jitter > 0 else 0.0
                        )
                        by = base_y + slope_y
                        if script_mode == "sup":
                            by -= base_size * float(script_height_offset)
                        elif script_mode == "sub":
                            by += base_size * max(0.06, float(script_height_offset) * 0.50)

                        has_f = "f" in tok and not tok.isupper()
                        has_desc = any(c in DESCENDERS for c in tok) and not tok.isupper()
                        is_punct = len(tok) == 1 and tok in SMALL_PUNCT
                        is_tall_bracket = len(tok) == 1 and tok in TALL_BRACKETS

                        if is_tall_bracket:
                            py = int(round(by - g_img.height * 0.88 + jitter_y))
                        elif has_f:
                            py = int(round(by - g_img.height * 0.78 + jitter_y))
                        elif has_desc:
                            py = int(round(by - g_img.height * 0.62 + jitter_y))
                        elif is_punct:
                            py = int(round(by - g_img.height + jitter_y * 0.25))
                        else:
                            py = int(round(by - g_img.height + jitter_y))

                        px = int(round(cursor_x))
                        if is_tall_bracket and px > margin_side:
                            px += max(2, abs(char_spacing) + 2)
                        pages[-1].paste(g_img, (px, py), g_img)
                        is_upper_tok = (len(tok) == 1 and tok.isupper()) or str(folder or '').startswith('upper_')
                        is_digit_tok = tok.isdigit() or (len(tok) == 1 and tok.isdigit())
                        if is_script:
                            eff_char_gap = max(-2, char_spacing) + 2
                        elif is_digit_tok:
                            eff_char_gap = max(4, char_spacing + 7)
                        elif is_upper_tok:
                            eff_char_gap = max(2, char_spacing + 5)
                        else:
                            eff_char_gap = min(char_spacing - 1, -2)
                        step_adv = max(5.0, g_img.width + eff_char_gap + rng.uniform(-0.3, 0.5))
                        if is_tall_bracket:
                            step_adv += 3.0
                        elif is_digit_tok:
                            step_adv += 3.0
                        elif is_upper_tok:
                            step_adv += 2.0
                        cursor_x += step_adv
                    else:
                        if tok == ".":
                            d_dot = ImageDraw.Draw(pages[-1])
                            dot_r = max(2, int(round(base_size * 0.055)))
                            dy_dot = int(round(base_y + (cursor_x - margin_side) * line_slope - dot_r))
                            dx_dot = int(round(cursor_x + dot_r))
                            d_dot.ellipse(
                                [dx_dot - dot_r, dy_dot - dot_r, dx_dot + dot_r, dy_dot + dot_r],
                                fill=seg_rgb,
                            )
                            cursor_x += dot_r * 2 + max(4, abs(char_spacing) + 3)
                        else:
                            cursor_x += max(6.0, base_size * lower_scale * 0.8 * len(tok) + char_spacing)

        if is_boxed_line:
         box_max_x = max(box_max_x, cursor_x)
         top_y = int(page_baselines[box_start_line] - base_size * 1.25)
         bot_y = int(page_baselines[line_idx] + max(18, int(base_size * 0.45)))
         box_right = min(PAGE_W - 20, int(max(box_max_x + 26, margin_side + 260)))
         draw_sketch_box(pages[-1], int(margin_side - 14), top_y, box_right, bot_y, default_rgb)
         
    _flush_src_line()
    globals()['_LAST_RENDER_HITBOXES'] = line_hitboxes
    return pages


def pages_to_pdf_bytes(pages: list[Image.Image]) -> bytes:
    buf = BytesIO()
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=rgb_pages[1:],
        resolution=150.0,
    )
    buf.seek(0)
    return buf.getvalue()


def pages_to_png_zip_bytes(pages: list[Image.Image]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, p in enumerate(pages, start=1):
            p_buf = BytesIO()
            p.convert("RGB").save(p_buf, format="PNG")
            zf.writestr(f"page_{idx:02d}.png", p_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()


def inline_multi_file_tex(tex_source: str, tex_map: dict[str, str], depth: int = 0) -> str:
    """Resolve \\input{...}, \\include{...}, and \\subfile{...} within a multi-file LaTeX project."""
    if depth > 6:
        return tex_source

    def _repl(match: re.Match) -> str:
        raw_target = match.group(1).strip()
        candidates = [
            raw_target,
            f"{raw_target}.tex",
            Path(raw_target).name,
            f"{Path(raw_target).name}.tex",
        ]
        for cand in candidates:
            if cand in tex_map:
                return "\n" + inline_multi_file_tex(tex_map[cand], tex_map, depth + 1) + "\n"
        return ""

    return re.sub(r"\\(?:input|include|subfile)\{([^{}]+)\}", _repl, tex_source)


def clean_latex_to_markup(tex_source: str) -> str:
    """Convert LaTeX source (including resume macros) into clean, readable handwriting text."""
    text = tex_source.replace("\r\n", "\n")
    # Strip comments (unescaped %)
    text = re.sub(r"(?<!\\)%.*", "", text)

    doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, flags=re.DOTALL)
    if doc_match:
        text = doc_match.group(1)

    # Remove preamble / layout / spacing directives
    text = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(
        r"\\(?:pagestyle|geometry|hypersetup|definecolor|newcommand|renewcommand|setlength|addtolength|titleformat|urlstyle|raggedbottom|raggedright|pdfgentounicode)\b[^\n]*",
        "",
        text,
    )
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.DOTALL)

    # Expand custom resume macros cleanly
    def _sub_heading(m: re.Match) -> str:
        org, loc, role, dates = (m.group(1).strip(), m.group(2).strip(), m.group(3).strip(), m.group(4).strip())
        line1 = f"{org} ({loc})" if loc else org
        line2 = f"{role} [{dates}]" if dates else role
        return f"\n{line1}\n{line2}\n"

    def _proj_heading(m: re.Match) -> str:
        title, yr = m.group(1).strip(), m.group(2).strip()
        return f"\n{title} ({yr})\n" if yr else f"\n{title}\n"

    text = re.sub(
        r"\\resumeSubheading\s*\{([^{}]*)\}\s*\{([^{}]*)\}\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        _sub_heading,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\resumeProjectHeading\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
        _proj_heading,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\\resumeItem\s*\{([^{}]*)\}",
        lambda m: f"\n- {m.group(1).strip()}",
        text,
        flags=re.DOTALL,
    )

    # Strip list/center wrappers and spacing commands like \vspace{2pt}
    text = re.sub(
        r"\\(?:begin|end)\{[A-Za-z0-9_*]+\}(?:\[[^\]]*\])?(?:\{[^{}]*\})*",
        "\n",
        text,
    )
    text = re.sub(r"\\rule(?:\[[^\]]*\])?\{[^{}]*\}\{[^{}]*\}", "\n", text)
    text = re.sub(r"\\(?:linewidth|textwidth|textheight|columnsep|parindent|parskip)\b", "", text)
    text = re.sub(r"\\(?:vspace|hspace)\*?\{[^}]*\}", " ", text)
    text = re.sub(
        r"\\(?:resumeSubHeadingListStart|resumeSubHeadingListEnd|resumeItemListStart|resumeItemListEnd)\b",
        "",
        text,
    )
    text = re.sub(
        r"\\(?:small|large|Large|LARGE|Huge|huge|normalsize|footnotesize|tiny|scshape|centering|noindent|hfill|vfill|medskip|bigskip|smallskip)\b",
        "",
        text,
    )

    # Unwrap common formatting & link commands while keeping inner text
    text = re.sub(r"\\href\{[^}]*\}\{([^{}]*)\}", r"\1", text)
    for _ in range(3):
        text = re.sub(
            r"\\(?:textbf|textit|textsf|texttt|underline|emph|mbox)\{([^{}]*)\}",
            r"\1",
            text,
        )

    text = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}", r"\n\n\1\n", text)
    text = re.sub(r"\\textcolor\{red\}\{([^{}]*)\}", r"[red]\1[/red]", text)
    text = re.sub(r"\\textcolor\{blue\}\{([^{}]*)\}", r"[blue]\1[/blue]", text)
    text = re.sub(r"\\textcolor\{green\}\{([^{}]*)\}", r"[green]\1[/green]", text)
    text = re.sub(r"\\fbox\{([^{}]*)\}", r"\n[box]\1[/box]\n", text)
    text = re.sub(r"\\item\b(?:\[[^\]]*\])?\s*", "\n- ", text)

    # Math delimiters
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: f"\n{m.group(1).strip()}\n", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: f"\n{m.group(1).strip()}\n", text, flags=re.DOTALL)
    text = re.sub(r"\$([^$\n]+)\$", r"\1", text)
    text = re.sub(r"\\\((.*?)\\\)", r"\1", text)

    # Protect \frac and \sqrt
    protected_tokens: list[str] = []
    text = re.sub(r"\\frac(?=\{)", lambda m: "@@FRAC_START@@", text)
    out_parts: list[str] = []
    idx_p = 0
    while idx_p < len(text):
        if text.startswith("@@FRAC_START@@", idx_p):
            after_cmd = idx_p + len("@@FRAC_START@@")
            if after_cmd < len(text) and text[after_cmd] == "{":
                num_s, after_num = extract_braced_arg(text, after_cmd)
                if after_num < len(text) and text[after_num] == "{":
                    den_s, end_pos = extract_braced_arg(text, after_num)
                    tok_id = f"ZFRAC{len(protected_tokens)}Z"
                    protected_tokens.append(f"\\frac{{{num_s}}}{{{den_s}}}")
                    out_parts.append(tok_id)
                    idx_p = end_pos
                    continue
        if text.startswith(r"\sqrt{", idx_p):
            inner_s, end_pos = extract_braced_arg(text, idx_p + len(r"\sqrt"))
            tok_id = f"ZSQRT{len(protected_tokens)}Z"
            protected_tokens.append(f"\\sqrt{{{inner_s}}}")
            out_parts.append(tok_id)
            idx_p = end_pos
            continue
        out_parts.append(text[idx_p])
        idx_p += 1
    text = "".join(out_parts).replace("@@FRAC_START@@", r"\frac")

    for cmd, tok_name in sorted(LATEX_MATH_TO_TOKEN.items(), key=lambda x: len(x[0]), reverse=True):
        sym = MATH_FALLBACK_UNICODE.get(tok_name, tok_name)
        text = re.sub(re.escape(cmd) + r"(?![A-Za-z])", f" {sym} ", text)

    for _ in range(4):
        text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r" \1 ", text)

    # Convert underscores to spaces so they don't render as stray bars/underscores
    text = text.replace(r"\_", " ").replace("_", " ")
    text = text.replace(r"\&", "&").replace(r"\%", "%").replace(r"\#", "#")
    text = text.replace(r"\\", "\n")
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    for i_tok, orig_expr in enumerate(protected_tokens):
        clean_expr = orig_expr.replace("_", " ")
        key_f = f"ZFRAC{i_tok}Z".upper()
        key_s = f"ZSQRT{i_tok}Z".upper()
        text = text.replace(key_f, clean_expr).replace(key_s, clean_expr)

    result = text.strip()
    if not result:
        fallback_lines = [ln.strip() for ln in tex_source.splitlines() if ln.strip() and not ln.strip().startswith("%")]
        return "\n".join(fallback_lines) if fallback_lines else tex_source.strip()
    return result

def smart_format_pasted_text(raw_str: str) -> str:
    """Clean up copy-pasted text or PDF dumps by preserving headings, numbered items, and questions."""
    txt = raw_str.replace("\r\n", "\n")
    # Insert line breaks before numbered items or Q.1) / 1) / 1. when run together inline
    txt = re.sub(r"(?<!^)(?<!\n)\s+(?=(?:Q(?:uestion)?\.?\s*\d+[\).:]|\d+[\).:]\s+[A-Z]))", "\n\n", txt)
    # Break after ALL-CAPS assignment/section titles followed by lowercase or mixed sentences
    txt = re.sub(r"\b([A-Z][A-Z\s]{8,45})\s+((?:Submission|Question|Topic|Unit|Chapter|Note|Instructions)\b)", r"\1\n\2", txt)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in txt.splitlines()]
    out_lines: list[str] = []
    for ln in lines:
        if not ln:
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines).strip()


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()

    if name.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(raw_bytes), "r") as zf:
            tex_entries = [n for n in zf.namelist() if n.lower().endswith(".tex") and not n.startswith("__MACOSX")]
            if not tex_entries:
                return "No .tex files found inside the uploaded ZIP."
            tex_map: dict[str, str] = {}
            for entry in tex_entries:
                content = zf.read(entry).decode("utf-8", errors="replace")
                tex_map[entry] = content
                tex_map[Path(entry).name] = content
            chosen = next(
                (n for n in tex_entries if Path(n).name.lower() == "main.tex"),
                max(tex_entries, key=lambda n: zf.getinfo(n).file_size),
            )
            combined_tex = inline_multi_file_tex(tex_map[chosen], tex_map)
            return clean_latex_to_markup(combined_tex)

    if name.endswith(".tex"):
        return clean_latex_to_markup(raw_bytes.decode("utf-8", errors="replace"))

    if name.endswith(".pdf"):
        page_texts: list[str] = []
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            for page in doc:
                # Extract text blocks in reading order to preserve lines & headings
                blocks = page.get_text("blocks")
                blocks_sorted = sorted(blocks, key=lambda b: (round(b[1] / 12.0), b[0]))
                b_lines: list[str] = []
                for b in blocks_sorted:
                    b_str = str(b[4]).strip()
                    if not b_str:
                        continue
                    for raw_l in b_str.splitlines():
                        cl = re.sub(r"[ \t]+", " ", raw_l).strip()
                        cl = re.sub(r"\s+([.,:;!?])", r"\1", cl)
                        if cl:
                            b_lines.append(cl)
                    b_lines.append("")
                page_texts.append("\n".join(b_lines).strip())
        return smart_format_pasted_text("\n\n".join(p for p in page_texts if p))

    return raw_bytes.decode("utf-8", errors="replace")


def analyze_missing_glyphs(text: str, bank: dict[str, list[Image.Image]]) -> tuple[list[str], list[str]]:
    """Return (missing_folders, single_sample_folders) referenced by the current editor text."""
    needed: set[str] = set()
    for raw_line in text.splitlines():
        if re.fullmatch(r"\[flow\].*?\[/flow\]", raw_line.strip(), flags=re.IGNORECASE):
            for f_node in ("flow_rect", "flow_oval", "flow_diamond", "flow_parallelogram", "flow_arr_right"):
                if f_node in bank:
                    needed.add(f_node)
            continue
        cleaned = re.sub(r"\[/?(?:black|blue|red|green|box)\]", "", raw_line)
        for word in cleaned.split():
            for chunk_str, _ in parse_script_chunks(word):
                for tok, folder in tokenize_word_with_ligatures(chunk_str, bank):
                    if folder:
                        needed.add(folder)

    missing = sorted(f for f in needed if not bank.get(f))
    single_sample = sorted(f for f in needed if len(bank.get(f, [])) == 1)
    return missing, single_sample


def render_draggable_split_workspace(
    pages: list[Image.Image],
    box_height: int = 760,
    preview_width_pct: int = 100,
    split_ratio_pct: int = 43,
    line_hitboxes: list[dict[str, int]] | None = None,
) -> None:
    img_tags: list[str] = []
    hb_list = line_hitboxes if line_hitboxes is not None else globals().get("_LAST_RENDER_HITBOXES", [])
    for idx, p in enumerate(pages, start=1):
        buf = BytesIO()
        p.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        page_boxes = [b for b in hb_list if int(b.get("page", 1)) == idx]
        boxes_json = json.dumps(page_boxes).replace('"', "&quot;")
        img_tags.append(
            f'<div style="margin-bottom:18px;text-align:center;">'
            f'<div style="font-size:12px;color:#475569;margin-bottom:4px;">Page {idx} <span style="color:#64748b;">(click any line)</span></div>'
            f'<img class="preview-page-img" data-page="{idx}" data-page-h="{p.height}" data-boxes="{boxes_json}" '
            f'src="data:image/png;base64,{b64}" '
            f'style="width:100%;height:auto;display:block;cursor:pointer;border:1px solid #cbd5e1;box-shadow:0 2px 8px rgba(0,0,0,0.08);" />'
            f"</div>"
        )
    all_imgs_html = "".join(img_tags)
    vp_h = max(340, int(box_height) - 16)
    init_zoom = max(50, min(160, int(preview_width_pct)))
    init_split = max(20, min(78, int(split_ratio_pct)))

    html = f"""
    <div id="zoom-viewport" style="width:100%;height:{vp_h}px;overflow:auto;resize:vertical;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;padding:12px;box-sizing:border-box;">
      <div id="sticky-zoom-bar" style="position:sticky;top:0;z-index:25;background:#f8fafc;padding:6px 4px 8px 4px;border-bottom:1px solid #e2e8f0;margin-bottom:10px;font-size:12px;color:#334155;display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
          <span><b>Ctrl+Scroll / Pinch</b> (<b id="zoom-readout">{init_zoom}%</b>)</span>
          <span>&bull;</span>
          <span>Split: <b>{init_split}%</b> / <b>{100 - init_split}%</b></span>
          <span id="clicked-line-badge" style="padding:2px 7px;border-radius:4px;background:#e2e8f0;color:#0f172a;font-weight:600;">Click a line</span>
        </div>
        <div style="display:flex;gap:4px;align-items:center;">
          <button type="button" id="zoom-out" style="padding:2px 9px;border:1px solid #94a3b8;border-radius:4px;background:#ffffff;color:#0f172a;cursor:pointer;font-weight:bold;">-</button>
          <button type="button" id="zoom-in" style="padding:2px 9px;border:1px solid #94a3b8;border-radius:4px;background:#ffffff;color:#0f172a;cursor:pointer;font-weight:bold;">+</button>
          <button type="button" id="reset-zoom" style="padding:2px 9px;border:1px solid #94a3b8;border-radius:4px;background:#ffffff;color:#0f172a;cursor:pointer;">Reset (100%)</button>
        </div>
      </div>
      <div id="zoom-stage" style="width:{init_zoom}%;margin:0 auto;">
        {all_imgs_html}
      </div>
    </div>
    <script>
      (function() {{
        const vp = document.getElementById("zoom-viewport");
        const stage = document.getElementById("zoom-stage");
        const readout = document.getElementById("zoom-readout");
        const badge = document.getElementById("clicked-line-badge");
        const btnIn = document.getElementById("zoom-in");
        const btnOut = document.getElementById("zoom-out");
        const btnReset = document.getElementById("reset-zoom");

        let zoom = {init_zoom};

        function applyZoom(newZoom) {{
          zoom = Math.max(45, Math.min(240, Math.round(newZoom)));
          if (stage) stage.style.width = zoom + "%";
          if (readout) readout.textContent = zoom + "%";
        }}

        if (btnIn) {{
          btnIn.addEventListener("click", function(e) {{
            e.preventDefault();
            e.stopPropagation();
            applyZoom(zoom + 15);
          }});
        }}
        if (btnOut) {{
          btnOut.addEventListener("click", function(e) {{
            e.preventDefault();
            e.stopPropagation();
            applyZoom(zoom - 15);
          }});
        }}
        if (btnReset) {{
          btnReset.addEventListener("click", function(e) {{
            e.preventDefault();
            e.stopPropagation();
            applyZoom(100);
          }});
        }}

        function onWheelZoom(e) {{
          if (e.ctrlKey || e.metaKey) {{
            e.preventDefault();
            e.stopPropagation();
            const step = e.deltaY < 0 ? 12 : -12;
            applyZoom(zoom + step);
          }}
        }}
        if (vp) {{
          vp.addEventListener("wheel", onWheelZoom, {{ passive: false, capture: true }});
        }}

        function jumpToEditorLine(srcLineIdx, clickRatioX) {{
          const lineNum = Math.max(1, (Number(srcLineIdx) || 0) + 1);
          const atEnd = (Number(clickRatioX) || 0) >= 0.62;
          if (badge) {{
            badge.textContent = "Line " + lineNum + (atEnd ? " (end)" : " (start)");
          }}
          try {{
            const pDoc = window.parent && window.parent.document ? window.parent.document : null;
            if (!pDoc) return;
            const ta = pDoc.querySelector("textarea");
            if (!ta) return;

            const fullVal = ta.value || "";
            const rawLines = fullVal.split("\\n");
            const idx = Math.max(0, Math.min(rawLines.length - 1, (Number(srcLineIdx) || 0)));

            let startPos = 0;
            for (let i = 0; i < idx; i++) {{
              startPos += rawLines[i].length + 1;
            }}
            const targetPos = atEnd ? (startPos + rawLines[idx].length) : startPos;

            ta.focus();
            if (typeof ta.setSelectionRange === "function") {{
              ta.setSelectionRange(targetPos, targetPos);
            }}
            ta.scrollTop = Math.max(0, idx * 22 - (ta.clientHeight || 240) * 0.35);
          }} catch (_) {{
            // If sandboxed, badge still highlights the selected line number
          }}
        }}

        if (stage) {{
          stage.querySelectorAll("img.preview-page-img").forEach(function(imgEl) {{
            imgEl.addEventListener("click", function(ev) {{
              const rect = imgEl.getBoundingClientRect();
              if (!rect.height || !rect.width) return;
              const pageH = Number(imgEl.dataset.pageH) || 1754;
              const relY = ((ev.clientY - rect.top) / rect.height) * pageH;
              const relX = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
              let boxes = [];
              try {{
                boxes = JSON.parse(imgEl.dataset.boxes || "[]");
              }} catch (_) {{
                boxes = [];
              }}

              let chosen = null;
              let bestDist = Infinity;
              for (const b of boxes) {{
                if (relY >= b.y0 - 12 && relY <= b.y1 + 12) {{
                  chosen = b;
                  break;
                }}
                const mid = (b.y0 + b.y1) * 0.5;
                const d = Math.abs(relY - mid);
                if (d < bestDist) {{
                  bestDist = d;
                  chosen = b;
                }}
              }}
              if (chosen && chosen.src_line !== undefined) {{
                jumpToEditorLine(chosen.src_line, relX);
              }} else {{
                const estLine = Math.max(0, Math.floor((relY - 120) / 56));
                jumpToEditorLine(estLine, relX);
              }}
            }});
          }});
        }}
      }})();
    </script>
    """
    components.html(html, height=int(box_height), scrolling=False)


st.set_page_config(page_title="Handwriting Glyph-Bank Studio", layout="wide")

if "latex_editor_box" in st.session_state and "How to use this studio" in str(st.session_state["latex_editor_box"]):
    st.session_state["latex_editor_box"] = (
        "this page is written from my own cropped handwriting samples.\n\n"
        "you can mix [blue]blue notes[/blue] and [red]red highlights[/red] on any line.\n\n"
        "equations: f(y) = y^2 + z^2 + a^2, \\frac{dy}{dx}"
    )

st.title("Handwriting Glyph-Bank Studio")
st.caption("Draggable split-pane live handwriting editor, 3-category template pack (Sheets 1–11), and PDF exporter.")

if "session_bank" not in st.session_state:
    st.session_state.session_bank = load_raw_glyph_bank(DEFAULT_GLYPHS_DIR)

bank: dict[str, list[Image.Image]] = st.session_state.session_bank
SIDEBAR_DEFAULTS: dict[str, object] = {
    "bg_mode": "Ruled White",
    "default_hex": "#181A1F",
    "opacity": 1.00,
    "stroke_delta": 0,
    "seed": 42,
    "base_size": 30,
    "prev_base_size": 30,
    "cap_scale": 1.06,
    "lower_scale": 0.53,
    "width_scale": 0.86,
    "line_stretch": 0.92,
    "script_scale": 0.75,
    "script_height_offset": 0.25,
    "char_spacing": -2,
    "word_spacing": 12,
    "line_gap": 64,
    "margin_side": 85,
    "baseline_jitter": 1,
    "tilt_deg": 0.7,
    "preview_box_h": 760,
    "preview_width_pct": 100,
    "split_ratio_pct": 43,
}


def init_sidebar_defaults() -> None:
    for k, v in SIDEBAR_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_preset_dict(cfg: dict[str, object]) -> None:
    for k, v in cfg.items():
        if k in st.session_state:
            st.session_state[k] = v
    if "base_size" in cfg:
        st.session_state["prev_base_size"] = cfg["base_size"]


def reset_ink_defaults() -> None:
    for k in ("bg_mode", "default_hex", "opacity", "stroke_delta"):
        st.session_state[k] = SIDEBAR_DEFAULTS[k]



def _sync_query_params(keys: tuple[str, ...]) -> None:
    try:
        qp = st.query_params
        for k in keys:
            if k in qp and k in st.session_state and not st.session_state.get(f"_qp_loaded_{k}"):
                raw_v = qp[k]
                cur_v = st.session_state[k]
                if isinstance(cur_v, int):
                    st.session_state[k] = int(raw_v)
                elif isinstance(cur_v, float):
                    st.session_state[k] = float(raw_v)
                elif isinstance(cur_v, str):
                    st.session_state[k] = str(raw_v)
                st.session_state[f"_qp_loaded_{k}"] = True
    except Exception:
        pass

def _save_query_params(keys: tuple[str, ...]) -> None:
    try:
        qp = st.query_params
        for k in keys:
            if k in st.session_state:
                qp[k] = str(st.session_state[k])
    except Exception:
        pass

def reset_proportion_defaults() -> None:
    for k in (
        "seed",
        "base_size",
        "prev_base_size",
        "cap_scale",
        "lower_scale",
        "width_scale",
        "line_stretch",
        "script_scale",
        "script_height_offset",
        "char_spacing",
        "word_spacing",
        "line_gap",
        "margin_side",
        "baseline_jitter",
        "tilt_deg",
    ):
        st.session_state[k] = SIDEBAR_DEFAULTS[k]
    st.session_state["char_scale_map"] = {}


def on_base_size_change() -> None:
    new_bs = int(st.session_state["base_size"])
    old_bs = int(st.session_state.get("prev_base_size", new_bs))
    if old_bs > 0 and new_bs != old_bs:
        ratio = new_bs / old_bs
        st.session_state["word_spacing"] = int(
            np.clip(round(int(st.session_state["word_spacing"]) * ratio), 10, 45)
        )
        st.session_state["line_gap"] = int(
            np.clip(round(int(st.session_state["line_gap"]) * ratio), 40, 76)
        )
        st.session_state["margin_side"] = int(
            np.clip(round(int(st.session_state["margin_side"]) * ratio), 60, 180)
        )
        st.session_state["baseline_jitter"] = int(
            np.clip(round(int(st.session_state["baseline_jitter"]) * ratio), 0, 4)
        )
    st.session_state["prev_base_size"] = new_bs


init_sidebar_defaults()

with st.sidebar:
    st.header("Glyph Bank (.zip & Local Folder)")
    total_crops = sum(len(v) for v in bank.values())
    st.info(f"Active folders: {len(bank)} | Total crops: {total_crops}")

    if bank:
        st.download_button(
            label="Download Glyph Bank (.zip)",
            data=build_zip_from_bank(bank),
            file_name="handwriting_glyphs.zip",
            mime="application/zip",
            key="sidebar_zip_dl",
        )
        if st.button("Save Bank to `glyphs/` Folder"):
            written = save_bank_to_folder(bank, DEFAULT_GLYPHS_DIR)
            st.success(f"Saved {written} glyph PNGs to `{DEFAULT_GLYPHS_DIR}`.")

    uploaded_zip = st.file_uploader("Load saved Glyph Bank (.zip)", type=["zip"])
    if uploaded_zip is not None and st.button("Load ZIP into Session"):
        loaded_zip_bank = load_bank_from_zip(uploaded_zip.getvalue())
        if loaded_zip_bank:
            st.session_state.session_bank = loaded_zip_bank
            st.success(f"Loaded {len(loaded_zip_bank)} folders from ZIP.")
            st.rerun()
        else:
            st.warning("No valid PNG glyph folders found inside that ZIP.")

    st.header("Style Presets")
    presets_map = load_saved_presets()
    chosen_preset = st.selectbox("Select a style preset", list(presets_map.keys()))
    if st.button("Apply Preset"):
        apply_preset_dict(presets_map[chosen_preset])
        st.rerun()
    new_preset_name = st.text_input("Save current settings as preset", placeholder="e.g. Lab Report Blue")
    if st.button("Save Preset") and new_preset_name.strip():
        cur_cfg = {
            k: st.session_state[k]
            for k in (
                "bg_mode",
                "default_hex",
                "opacity",
                "stroke_delta",
                "base_size",
                "cap_scale",
                "lower_scale",
                "char_spacing",
                "word_spacing",
                "line_gap",
                "margin_side",
                "baseline_jitter",
                "tilt_deg",
            )
        }
        persist_custom_preset(new_preset_name.strip(), cur_cfg)
        st.success(f"Saved preset '{new_preset_name.strip()}'.")
        st.rerun()

    st.header("Background & Ink")
    st.button("Reset Ink & Background to Default", on_click=reset_ink_defaults, key="btn_reset_ink")
    bg_mode = st.selectbox(
        "Page Background",
        ["Plain White", "Ruled White", "Beige Ruled", "Uploaded Image"],
        key="bg_mode",
    )
    custom_bg: Image.Image | None = None
    if bg_mode == "Uploaded Image":
        bg_file = st.file_uploader("Upload background / notebook image", type=["png", "jpg", "jpeg", "webp"])
        if bg_file is not None:
            custom_bg = Image.open(bg_file).convert("RGB").copy()

    default_hex = st.color_picker("Default Ink Color", key="default_hex")
    default_rgb = hex_to_rgb(default_hex)
    opacity = st.slider("Ink Opacity", 0.20, 1.00, step=0.02, key="opacity")
    stroke_delta = st.slider("Stroke Thickness Adjust", -2, 3, step=1, key="stroke_delta")

    st.header("Proportions & Density")
    cap_scale = float(st.session_state.get("cap_scale", 1.06))
    lower_scale = float(st.session_state.get("lower_scale", 0.53))
    width_scale = float(st.session_state.get("width_scale", 0.86))
    line_stretch = float(st.session_state.get("line_stretch", 0.90))
    st.button(
        "Reset Proportions & Density to Default",
        on_click=reset_proportion_defaults,
        key="btn_reset_props",
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=999999, step=1, key="seed")
    base_size = st.slider("Base Size (px)", 28, 72, step=1, key="base_size", on_change=on_base_size_change)
    script_scale = st.slider("Script Size (Sub/Super)", 0.40, 1.15, step=0.02, key="script_scale")
    script_height_offset = st.slider("Script Height Offset", 0.05, 0.55, step=0.01, key="script_height_offset")
    char_spacing = st.slider("Letter Density / Spacing (px)", -14, 12, step=1, key="char_spacing")

    if "char_scale_map" not in st.session_state:
        st.session_state["char_scale_map"] = {}
    with st.expander("Fix Out-of-Proportion Character"):
        available_keys = sorted(bank.keys()) if bank else ["a", "e", "i", "o", "s", "t", "y", "z"]
        target_ch = st.selectbox("Character / Folder", available_keys, key="fix_ch_select")
        cur_val = float(st.session_state["char_scale_map"].get(target_ch, 1.00))
        adj_val = st.slider(
            f"Scale for `{target_ch}`",
            min_value=0.50,
            max_value=1.80,
            value=round(cur_val, 2),
            step=0.02,
            key="fix_ch_slider",
        )
        c_ov1, c_ov2 = st.columns(2)
        with c_ov1:
            if st.button("Apply Fix", key="apply_ch_fix"):
                st.session_state["char_scale_map"][target_ch] = float(adj_val)
                st.rerun()
        with c_ov2:
            if st.button("Clear Overrides", key="clear_ch_fix"):
                st.session_state["char_scale_map"] = {}
                st.rerun()
        if st.session_state["char_scale_map"]:
            st.caption("Active overrides: " + ", ".join(f"{k}={v:.2f}" for k, v in st.session_state["char_scale_map"].items()))
    word_spacing = st.slider("Word Spacing (px)", 10, 45, step=1, key="word_spacing")
    line_gap = st.slider("Line Gap (px)", 40, 76, step=1, key="line_gap")
    margin_side = st.slider("Side Margin (px)", 60, 180, step=5, key="margin_side")
    baseline_jitter = st.slider("Baseline Jitter (px)", 0, 4, step=1, key="baseline_jitter")
    tilt_deg = st.slider("Max Tilt (deg)", 0.0, 3.0, step=0.2, key="tilt_deg")

tab_render, tab_template, tab_recheck = st.tabs(
    ["1. Live Editor & PDF", "2. Template Sheets (1–11) & Upload", "3. Recheck, Coverage & Edit Glyphs"]
)

with tab_render:
    cur_split = int(st.session_state.get("split_ratio_pct", 43))
    left_ratio = max(0.20, cur_split / 100.0)
    right_ratio = max(0.20, (100 - cur_split) / 100.0)
    col_editor, col_preview = st.columns([left_ratio, right_ratio], gap="small")
    with col_editor:
        st.subheader("Write or Upload Text")
        st.caption(
            "Use `[red]text[/red]`, `[blue]text[/blue]`, `[green]text[/green]`, or `[black]text[/black]` for ink colors; "
            "`y^2`, `z^2`, `a^2`, `x_{i+1}`, `\\frac{dy}{dx}`, `\\sqrt{x_{i+1}}` for math; and `[flow]Start -> Input x_i -> ?Valid -> [blue]Process[/blue] -> End[/flow]` for flowcharts."
        )
        uploaded_doc = st.file_uploader("Upload `.tex`, Overleaf `.zip`, `.txt`, or `.pdf`", type=["tex", "zip", "txt", "pdf"])
        default_text = 'This page is written from my own cropped handwriting samples.\n\nYou can mix [blue]blue notes[/blue] and [red]red highlights[/red] on any line.\n\nequations: f(y) = y^2 + z^2 + a^2, \\frac{dy}{dx}'

        if "latex_editor_box" not in st.session_state:
            st.session_state["latex_editor_box"] = default_text

        if uploaded_doc is not None and st.button("Reload from Uploaded File"):
            extracted = extract_text_from_upload(uploaded_doc)
            st.session_state["latex_editor_box"] = extracted if extracted.strip() else default_text
            st.session_state["last_uploaded_doc_sig"] = f"{uploaded_doc.name}_{uploaded_doc.size}"
            st.rerun()

        if uploaded_doc is not None:
            file_sig = f"{uploaded_doc.name}_{uploaded_doc.size}"
            if st.session_state.get("last_uploaded_doc_sig") != file_sig:
                extracted = extract_text_from_upload(uploaded_doc)
                st.session_state["latex_editor_box"] = extracted if extracted.strip() else default_text
                st.session_state["last_uploaded_doc_sig"] = file_sig
                st.rerun()
        else:
            st.session_state.pop("last_uploaded_doc_sig", None)

        if st.button(
            "Clean Pasted LaTeX in Editor",
            help="Strips LaTeX commands (\\documentclass, \\usepackage, \\section, \\textbf, \\item) in the box below",
        ):
            st.session_state["latex_editor_box"] = clean_latex_to_markup(st.session_state["latex_editor_box"])
            st.rerun()

        input_text = st.text_area(
            "Type or paste text / LaTeX here (Ctrl+Scroll to resize font; Ctrl+Enter or click outside to refresh preview)",
            key="latex_editor_box",
            height=460,
        )
        live_preview = st.checkbox("Live auto-preview", value=True)
        manual_trigger = st.button("Generate Handwritten Pages", type="primary")

    with col_preview:
        st.subheader("Live Handwriting Preview")
        if live_preview or manual_trigger:
            if not bank:
                st.error("Load or extract at least a few character glyphs first (Tab 2 or sidebar ZIP).")
            elif not input_text.strip():
                st.warning("Type or upload some text on the left to preview.")
            else:
                pages = render_handwritten_pages(
                    text=input_text,
                    raw_bank=bank,
                    seed=int(seed),
                    base_size=base_size,
                    cap_scale=cap_scale,
                    lower_scale=lower_scale,
                    width_scale=width_scale,
                    line_stretch=line_stretch,
                    script_scale=script_scale,
                    script_height_offset=script_height_offset,
                    char_scale_map=st.session_state.get("char_scale_map", {}),
                    char_spacing=char_spacing,
                    word_spacing=word_spacing,
                    line_gap=line_gap,
                    margin_side=margin_side,
                    baseline_jitter=baseline_jitter,
                    tilt_deg=tilt_deg,
                    opacity=opacity,
                    stroke_delta=stroke_delta,
                    default_rgb=default_rgb,
                    bg_mode=bg_mode,
                    custom_bg=custom_bg,
                )
                usable_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
                eff_gap = max(line_gap, int(round(base_size * 1.65)))
                lines_per_page = max(1, usable_h // eff_gap)
                st.caption(f"Rendered {len(pages)} page(s) • Capacity: ~{lines_per_page} baselines/page at line gap {eff_gap}px.")

                pdf_bytes = pages_to_pdf_bytes(pages)
                png_zip_bytes = pages_to_png_zip_bytes(pages)
                c_dl1, c_dl2 = st.columns(2)
                with c_dl1:
                    st.download_button(
                        label=f"Download PDF ({len(pages)} page{'s' if len(pages) > 1 else ''})",
                        data=pdf_bytes,
                        file_name="handwritten_output.pdf",
                        mime="application/pdf",
                    )
                with c_dl2:
                    st.download_button(
                        label=f"Download All {len(pages)} Page PNGs (.zip)",
                        data=png_zip_bytes,
                        file_name="handwritten_pages_png.zip",
                        mime="application/zip",
                    )
                with st.expander("Preview Plane & Split Size Adjuster", expanded=False):
                    c_sz1, c_sz2, c_sz3 = st.columns(3)
                    with c_sz1:
                        st.slider(
                            "Preview Pane Height (px)",
                            min_value=380,
                            max_value=1300,
                            step=20,
                            key="preview_box_h",
                        )
                    with c_sz2:
                        st.slider(
                            "Preview Page Width (%)",
                            min_value=55,
                            max_value=145,
                            step=5,
                            key="preview_width_pct",
                        )
                    with c_sz3:
                        st.slider(
                            "Editor Column Split (%)",
                            min_value=20,
                            max_value=78,
                            step=1,
                            key="split_ratio_pct",
                        )
                render_draggable_split_workspace(
                    pages,
                    box_height=int(st.session_state.get("preview_box_h", 760)),
                    preview_width_pct=int(st.session_state.get("preview_width_pct", 100)),
                    split_ratio_pct=int(st.session_state.get("split_ratio_pct", 43)),
                    line_hitboxes=_LAST_RENDER_HITBOXES,
                )

with tab_template:
    st.subheader("Step A: Choose Template Category & Download Sheets")
    selected_cat = st.radio(
        "Template Sheet Category",
        options=list(SHEET_CATEGORIES.keys()),
        horizontal=True,
    )
    cat_sheets = SHEET_CATEGORIES[selected_cat]
    cat_slug = (
        "english_sheets"
        if selected_cat.startswith("1.")
        else ("math_sheets" if selected_cat.startswith("2.") else "flowchart_sheets")
    )

    st.download_button(
        label=f"Download All {len(cat_sheets)} Sheets in '{selected_cat}' (.zip)",
        data=build_template_pack_zip(cat_sheets, subfolder_name=cat_slug),
        file_name=f"{cat_slug}_pack.zip",
        mime="application/zip",
    )

    st.divider()
    preview_sheet_num = st.selectbox(
        "Inspect or download an individual sheet from this category",
        options=cat_sheets,
        index=0,
        format_func=lambda s: SHEET_INFO[s][0],
    )
    selected_title, selected_desc = SHEET_INFO[int(preview_sheet_num)]
    st.info(f"**{selected_title}:** {selected_desc}")

    single_sheet_img = build_template_sheet(int(preview_sheet_num))
    single_buf = BytesIO()
    single_sheet_img.save(single_buf, format="PNG")
    st.download_button(
        label=f"Download Sheet {int(preview_sheet_num):02d} Only (PNG)",
        data=single_buf.getvalue(),
        file_name=f"template_sheet_{int(preview_sheet_num):02d}.png",
        mime="image/png",
    )
    st.image(single_sheet_img, caption=selected_title, use_container_width=True)

    st.divider()
    st.subheader("Step B: Upload Any Filled Sheet(s) & Extract Glyphs")
    st.caption(
        "Upload any mix of filled English (01–05), Math (06–09), or Flowchart (10–11) sheets—each sheet is deskewed, "
        "auto-detected from its 4-bit corner marker, and denoised automatically."
    )
    filled_uploads = st.file_uploader(
        "Upload one or more filled PNG/JPG template sheets",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    merge_mode = st.radio("Import mode", ["Merge into current bank", "Replace current bank"], horizontal=True)

    if filled_uploads and st.button("Extract Glyphs from Uploaded Sheet(s)", type="primary"):
        combined_extracted: dict[str, list[Image.Image]] = {}
        detected_summary: list[str] = []
        for f_up in filled_uploads:
            pil_filled = Image.open(f_up)
            auto_sheet_num = detect_sheet_index(pil_filled, filename=f_up.name)
            detected_summary.append(f"{f_up.name} → Sheet {auto_sheet_num:02d}")
            part = extract_from_template_image(pil_filled, sheet_index=auto_sheet_num)
            for k, v in part.items():
                combined_extracted.setdefault(k, []).extend(v)

        if not combined_extracted:
            st.warning("No dark strokes detected inside the grid boxes.")
        else:
            if merge_mode == "Replace current bank":
                st.session_state.session_bank = combined_extracted
            else:
                for k, v in combined_extracted.items():
                    st.session_state.session_bank.setdefault(k, []).extend(v)
            st.session_state.last_extracted_count = sum(len(v) for v in combined_extracted.values())
            st.session_state.last_detected_sheets = ", ".join(detected_summary)
            st.rerun()

    if st.session_state.get("last_extracted_count"):
        det_info = st.session_state.get("last_detected_sheets", "")
        st.success(f"Extracted {st.session_state.last_extracted_count} glyph crops ({det_info})! Download your updated ZIP below or review in Tab 3.")

        if st.session_state.session_bank:
            st.download_button(
                label="Download Updated Glyph Bank (.zip)",
                data=build_zip_from_bank(st.session_state.session_bank),
                file_name="handwriting_glyphs.zip",
                mime="application/zip",
                key="tab2_zip_dl",
            )

with tab_recheck:
    st.subheader("Glyph Coverage Report for Current Editor Text")
    editor_snapshot = st.session_state.get("latex_editor_box", "")
    missing_list, single_list = analyze_missing_glyphs(editor_snapshot, bank)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("**Missing Glyph Folders (in current text):**")
        st.write(", ".join(f"`{m}`" for m in missing_list) if missing_list else "None — all characters in text have at least 1 crop.")
    with col_c2:
        st.markdown("**Single-Sample Folders (only 1 crop):**")
        st.write(", ".join(f"`{s}`" for s in single_list) if single_list else "None.")

    st.divider()
    st.subheader("Inspect, Delete Bad Crops, or Move a Misplaced Crop")
    if not bank:
        st.warning("No glyphs loaded yet. Extract from a template in Tab 2, load a ZIP in the sidebar, or place folders in `glyphs/`.")
    else:
        selected_folder = st.selectbox("Select character / ligature folder to inspect", sorted(bank.keys()))
        variants = bank.get(selected_folder, [])
        st.write(f"Current folder: `{selected_folder}` ({display_label(selected_folder)}) — {len(variants)} image(s)")

        if variants:
            cols = st.columns(min(6, len(variants)))
            for idx, im in enumerate(variants):
                with cols[idx % len(cols)]:
                    st.image(im, caption=f"Image #{idx + 1}", width=90)
                    if st.button("Delete", key=f"del_{selected_folder}_{idx}"):
                        variants.pop(idx)
                        if not variants:
                            del bank[selected_folder]
                        st.rerun()

            st.divider()
            st.markdown("**Fix a mis-sorted image (move it to the right character or ligature):**")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                move_idx = st.number_input("Which image number (#) to move?", min_value=1, max_value=len(variants), value=1, step=1)
            with col_m2:
                target_folder = st.text_input(
                    "Move it to which folder? (e.g. `a`, `upper_F`, `lig_pp`, `lig_and`, `question`)",
                    value=selected_folder,
                )
            if st.button("Move Image to Target Folder") and target_folder.strip():
                clean_target = target_folder.strip()
                if clean_target != selected_folder:
                    item = variants.pop(int(move_idx) - 1)
                    if not variants:
                        del bank[selected_folder]
                    bank.setdefault(clean_target, []).append(item)
                    st.success(f"Moved Image #{move_idx} to `{clean_target}`.")
                    st.rerun()