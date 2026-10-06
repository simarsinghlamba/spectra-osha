# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 109770, "status": "ok", "timestamp": 1791284306721, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="pourJcwDpgf8" outputId="0c31ac13-d494-41aa-bd76-a5b407f0317e"
# !pip -q install -U transformers accelerate safetensors huggingface_hub statsmodels
from google.colab import drive, userdata
drive.mount("/content/drive")
import sys; sys.path.insert(0, "/content/drive/MyDrive/spectra")
from spectra.common import *
from spectra.stats import point_biserial, bh
from spectra.sae import GemmaReader, load_sae, load_acts
from spectra.data import LEAK_RE
setup_dirs(); seed_all()
import joblib, numpy as np, pandas as pd, torch
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN").strip())

train, test = load_split("train"), load_split("test")
lm = load_json(DATA / "label_map.json"); classes, titles = lm["classes"], lm["titles"]
L = CFG["sae_main_layer"]
S_tr = load_acts(ACTS / "raw_train", "saemax", L)
S_te = load_acts(ACTS / "raw_test", "saemax", L)
freq = np.asarray((S_tr > 0).mean(0)).ravel()
top_classes = [c for c in train.y_event.value_counts().index if c != "other"][:8]
print("top classes:", [(c, titles[c].strip()) for c in top_classes])

disc = []
def add_target(name, X, y, k):
    r, p = point_biserial(X, y)
    r = np.where(freq >= 0.005, r, 0)
    for fid in np.argsort(-r)[:k]:
        disc.append(dict(target=name, feature=int(fid), r=float(r[fid]), p=float(p[fid]), freq=float(freq[fid])))
for c in top_classes:
    add_target(f"event:{c}", S_tr, train.y_event.values == c, 5)
n = CFG["masked_train_n"]
add_target("amputation:raw", S_tr[:n], train.amputation_bin.values[:n], 10)
add_target("amputation:masked", load_acts(ACTS / "masked_train", "saemax", L)[:n], train.amputation_bin.values[:n], 10)
disc = pd.DataFrame(disc); disc["p_bh"] = bh(disc.p.values)
save_table(disc, "discovery_top_features")
cand = np.array(sorted(disc.feature.unique())); print(len(cand), "candidate features")
amp_raw = set(disc[disc.target == "amputation:raw"].feature); amp_msk = set(disc[disc.target == "amputation:masked"].feature)
print("amputation features shared raw/masked:", len(amp_raw & amp_msk), "of 10")

# %% colab={"base_uri": "https://localhost:8080/", "height": 800, "referenced_widgets": ["98aa1c91618e44ab9533d893290a4740", "fa7684831c1142219656fe9957e56412", "8c0ea2618f5740fdb707008acc41db73", "7050b429cc704ab4af2413d632707c7a", "6c4b4ad63163474cb5f82734431c85da", "25a5368e58ca4ac39e08424ef2ffdba2", "3f0820f6f26949018f5bdfba84f42fc3", "c80126bd8fa140e291b0c6a66869018c", "d3581add36104561a515ce6bb597a182", "3089e5554acd481096292a2e29a4da72", "24b370897fd54b589881b02fe4d49276", "c3237de6a06a49bda487560499971ee4", "2002b5d018e3490096b229ce125eed0c", "2bb3693468374a7aa670909951f208a7", "9e23b64714084d429052212cfbf66b0e", "bb37e502b482418099d666be44198e89", "e5ed84d478af4922ae6bc19575860b6f", "e6628eb941f2414a96b7dba6760c53b8", "02218cbf3d094e2e855af9e8c0ed4f5e", "a41bae38d1e54ddaab903ca92b35eed7", "0cb5ceb7db5e4a5cbb67e2bf690fdaa7", "a989b5dcf1ef4ee3978d321aa9b02570", "4bb7ec9673104c1d8003c657fa527be4", "9446f1bd1e524dbea231254042425a24", "c1782a0b59cd41d582492cfaa005e525", "2866be7ae4644c749bb56dfcbe404cff", "b7c1442f3e0e4c9d99d30345a00e88bb", "e6b83bcd6eb94d33a5419f64d0c39a50", "8a6f88b89c1d49689e8d6149b8b61d5c", "12daf605cee14853ae92680140ff360a", "690107c3b9154cf2bdedcdcd264fc7c7", "3750c0dbbace473895ec2f7d53005f90", "233037597d36459fb500cfe7de186a39", "03d18dd0fceb484da54f290c4020f9f1", "857470cff79f43229cd9c739c77f9dbe", "f306be1c1b314d39950adf5e06094123", "10c73fcee9df4e4b94d9f1b73dc633c4", "be013ff9ba134ac7846b2d7f4bb4ef59", "e05fd449138b479cb5914e4105231839", "b92f7ccace3b4a65862c81f1b316648c", "ff3e8221798b44ddb5c1bb5cf525d22b", "27c0afc559df40579aa030be91287e3c", "07c8ae121ae04ac9b446a62d9175980a", "4f041c4b1f5047498b266bfddb52e9fa", "8048cf1ae0454114be987c0d4d8dfc4e", "498559c0c9034e0da30993003cf12c10", "b8d9ac066cc347d6b4db15bc7d914787", "4743adb98f294d5b93af822175541512", "ecea7cff583e4e2aa5ebb1b19bc88601", "eceb73a9ad194911add3a30d624c03cc", "5a5e7a4a00714bf0834e6c0b447b9d77", "3ef6c91e4bd74361b5c3bf16a5bd0061", "998af2c7dc8944489d13e029725a7aba", "0e89104af3a2489485159ed7d8307635", "52ae5c6907ac4347856268b89969eb67", "cda35db159ee430ca88bca62a08c9ba7", "0ddc3006969e476c893ce6de4025d0af", "fbf0a4234b4441159ccccd2d61453cc6", "c876de5fea5d46cdb2facd3e6958a7d7", "a23e4903030041f49d5c07f80aa9b302", "bfff2ee505b04272b0464235643434d4", "6856ffaa4890469595cde9da1876f766", "c81861a15f97431dba65341370121bd4", "d73016374366465dafdf59b333b5a5f1", "d6acdd0b3bbd4460be6c99fd32af6c26", "3ce119c4fbf248cc80029fe6356dcfe6", "8fe8301b79b1438c9fca594807c91fdb", "2d5dbee6c83d4e4eb32179a18da4b5e2", "0ce53eb3195644819e2e6d9d1e3b45cc", "911f85cee5984069a362f6481e2e8b3f", "ba44b874436e4f2eb666d9bfeb0e819c", "fafb94d500344cefbf9eb14045f55dfa", "36eb2834657941f5ae57ec5618fd1e48", "6ac6d10d2542485b8b052d6247c78840", "8cac1fda39314c11bccfb0c6342b12e2", "73049b97f7b848a687a96950a0bbafe2", "f013442e627f43dbab157ac1a2232cd9", "f0511f2d149a4692a233eb88533c0a6c", "c42fdd7eedc0408ba91e9f8bae7a7d85", "8d7f27cb63a34a16a36534a1f37457b2", "93faaa71c8d346c2903f573b991d7513", "b03af093063644be93ffeb9cc7c0304e", "6f80bf1c70344edab813152f37733ed8", "ff30b07f38fb471f844c994f3d85532f", "d22774686e374318a40a803e193e764c", "7f853733a0be44cf95e2702818e26f92", "6f55281bc913459bac78f89b8ff424d6", "a9baede9f2d04713a151bf7c72e25b79", "4234a72bf37b4ddfbc73df078f1eb940", "f480dc6207ec4d2da339ab7998b6a97a", "fa58ee1224094f85aac382a4cd01d6ae", "3cb72909e53e410697f3e391325e9774", "dcf4d552e9b24924b41306fbbc6f2043", "2afeb32b27c04524a26a2ec7d20e6a9f", "056863eca99b4de6bc41236ed45f2f6a", "55c2c7abe4db4d0d8a7147214794004b", "65b9b340a19e4874adcea2654be32401", "0355b27eb44542e2877155f2c12c9219", "68cd37cbd46240f4b936ec97cb3c0958", "a4cb31d863c34e6f8aa60c74650f1e32", "201dc90f44ec47738c7ccb6c8d56387e", "c03804dc22be461c8354a387a39f20b0", "4ea1132e696243bdb865c0c0e62552e9", "0d12449c88b74e4a98b7698378a68058", "4dcf641bba8c4d619286f75a3c59a0b4", "1fc3fc3fa54a4cb598a548741b1b1f86", "06bb07c70fda4b049bc7ce4b12da2aa4", "09a4816df6a54e9aa6e98a8d0801575f", "d1cbea149859434abacded4be305b027", "6f939f726a1c456ab32da9dae1d37186", "5f38f125bfa340928ff484cd2b328157", "3bef061da749449f8ca4d5e38152e067", "0796a240784c40728977a723dfa6884d", "8f0167b450aa472280c296a6a3a5380b", "746d4a20c8b84aed9f7f54d18655a368", "0a82acef214146479870124634149a3b", "b334ab4a85774f7c908a5dd2ade4b2a5", "9279c4faea874dc18a98b54961b7e5e7", "696a6017fe15455eac848181850793e1", "6086f628d3f94e178d1cd798a052ce7a", "55022c6df0654ad1a5f433d842c7cfc1", "0df666b036c449008247e0ff6d4c344c", "703322f8fa4c482d9f5c1868cf11859d", "e9ae29930f2b4c029ca33e98f95f1153", "dcc6d303c8ea494c91f89c9bf96630c1", "80c2b86b2cd242b99edbdb643490cd4a", "4ad6f3f203ab4ce580e8aa145c8e9b18", "21ee5c57518244a89dcade84d4715d52", "0dcc8435094f40cd926e0f75154bd539", "19d0b1e63f4a41d4ad88685719c31942", "c6048d1b5ab440c9b35fa64badbb2737", "3301e1e2ff064e489e7012443fd98471", "c909b6c96bb748bf9c09aea1a18211d0", "58400fd7ba6545fea8aa7b31f65746f4", "428eb7e3b98642ffb6b335c3056f0112", "ec09a67b97fc461a90b70689ab867b1c", "d3022e5634f143709107243bf642fdad", "9815efd048a045e58ede8488c57b7342", "c61f442fad964375b796f2690b63069c", "85a37f2b3abb4cbc93acd881bb401445", "ab0f56ebf69d45e5a9cffab56958d65e", "96b4e002563544bd842044c8df39e628", "c71e5d3901d7451a9b733165127234bc"]} executionInfo={"elapsed": 109598, "status": "ok", "timestamp": 1791284420651, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="14POQjT2p0VQ" outputId="48b6a918-c416-4691-f14a-3c483ccce304"
reader = GemmaReader(CFG["gemma"], [13, 17], max_len=CFG["max_len"])
sae13, _ = load_sae(13, CFG["sae_width"], CFG["sae_l0"], CFG["sae_repo"], reader.d_model)
sample = test.sample(3000, random_state=SEED).reset_index(drop=True)
texts, K, TOP = sample.narrative.tolist(), len(cand), 10
cand_t = torch.tensor(cand, device="cuda")
best_v, best_d, best_p = np.full((K, TOP), -1.0), np.zeros((K, TOP), int), np.zeros((K, TOP), int)
ids_store = {}
for b0 in range(0, len(texts), 32):
    cache, mask, ids = reader.run(texts[b0:b0 + 32], stop_at=13)
    f = (sae13.encode(cache[13].float()) * mask.unsqueeze(-1))[..., cand_t]
    v, pos = f.max(1); v, pos = v.cpu().numpy().T, pos.cpu().numpy().T
    for j in range(ids.shape[0]):
        ids_store[b0 + j] = ids[j][: int(mask[j].sum()) + 1].cpu().numpy()
    docs = np.tile(np.arange(b0, b0 + ids.shape[0]), (K, 1))
    allv, alld, allp = (np.concatenate(a, 1) for a in ((best_v, v), (best_d, docs), (best_p, pos)))
    o = np.argsort(-allv, 1)[:, :TOP]
    best_v, best_d, best_p = (np.take_along_axis(a, o, 1) for a in (allv, alld, allp))

dec = lambda a: reader.tok.decode(a)
def snippet(doc, pos, w=10):
    t = ids_store[doc]; s, e = max(1, pos - w), min(len(t), pos + w + 1)
    return f"{dec(t[s:pos])} [[{dec(t[pos:pos + 1]).strip()}]] {dec(t[pos + 1:e])}".strip()
def near_leak(doc, pos):
    t = ids_store[doc]; return bool(LEAK_RE.search(dec(t[max(1, pos - 2):pos + 3])))

cards = {}
for k, fid in enumerate(cand):
    good = best_v[k] > 0
    snips = [snippet(d, p) for d, p in zip(best_d[k][good], best_p[k][good])]
    leak = float(np.mean([near_leak(d, p) for d, p in zip(best_d[k][good], best_p[k][good])])) if good.any() else 0.0
    cards[int(fid)] = dict(feature=int(fid), layer=L, snippets=snips, leak_share=leak,
                           targets=disc[disc.feature == fid].target.tolist(),
                           r_max=float(disc[disc.feature == fid].r.max()), freq=float(freq[fid]))
save_json(cards, WORK / "cards_raw.json")

lab = pd.DataFrame([{**{k: v for k, v in c.items() if k != "snippets"}, "targets": "; ".join(c["targets"]),
                     **{f"snippet_{i + 1}": s for i, s in enumerate(c["snippets"])}, "label": "", "rating": ""}
                    for c in cards.values()])
lab.to_csv(REPO / "results" / "feature_labeling.csv", index=False)

# shortcut evidence: how often do amputation features fire right on an outcome word?
for tgt in ["amputation:raw", "amputation:masked"]:
    fs = disc[disc.target == tgt].feature
    print(f"{tgt}: mean leak_share = {np.mean([cards[int(f)]['leak_share'] for f in fs]):.2f} | "
          f"features firing on outcome words (>=50%): {sum(cards[int(f)]['leak_share'] >= 0.5 for f in fs)}/10")
for fid in list(disc[disc.target == "amputation:raw"].feature)[:3]:
    print(f"\n#{fid} (leak {cards[int(fid)]['leak_share']:.0%}):", *cards[int(fid)]["snippets"][:3], sep="\n  ")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2295325, "status": "ok", "timestamp": 1791286795570, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="e-5rc9im20qg" outputId="0da29a2b-da1c-476e-ebfb-f54f5ae01d97"
N_ROWS, N_RAND = 80, 50          # sized for a free T4 (~2 min per feature); raise if time allows
probe17 = joblib.load(MODELS / "probe_dense_L17_event.joblib")
cls_index = {c: i for i, c in enumerate(probe17.classes_)}

def p_true(texts, y, feat=None, bs=40):
    out = []
    reader.edit = None if feat is None else (13, lambda h, f=int(feat): h - (sae13.encode_one(h, f) * reader.mask).unsqueeze(-1) * sae13.W_dec[f])
    for b0 in range(0, len(texts), bs):
        cache, mask, _ = reader.run(texts[b0:b0 + bs], stop_at=17)
        m = mask.unsqueeze(-1).float()
        pooled = ((cache[17].float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
        P = probe17.predict_proba(pooled)
        out.append(P[np.arange(len(P)), [cls_index[v] for v in y[b0:b0 + bs]]])
    reader.edit = None
    return np.concatenate(out)

rng = np.random.default_rng(SEED)
tests_ = [(c, int(fid)) for c in top_classes
          for fid in disc[disc.target == f"event:{c}"].sort_values("r", ascending=False).feature.head(2)]
abl = []
for c, fid in tests_:
    rows_c = np.where((test.y_event.values == c) & (S_te[:, fid].toarray().ravel() > 0))[0][:N_ROWS]
    if len(rows_c) < 20:
        print("skip", c, fid, "too few active narratives"); continue
    texts_c, y_c = test.narrative.values[rows_c].tolist(), test.label_id.values[rows_c]
    base = p_true(texts_c, y_c)
    eff = float((base - p_true(texts_c, y_c, fid)).mean())
    active = np.asarray((S_te[rows_c] > 0).mean(0)).ravel()
    pool = np.setdiff1d(np.where(active >= 0.10)[0], [fid])
    rand = rng.choice(pool, size=min(N_RAND, len(pool)), replace=False)
    null = np.array([(base - p_true(texts_c, y_c, r)).mean() for r in rand])
    p = (1 + np.sum(null >= eff)) / (len(null) + 1)
    abl.append(dict(cls=c, title=titles[c].strip(), feature=fid, n=len(rows_c), effect=eff,
                    null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                    ratio_vs_random=float(eff / (np.abs(null).mean() + 1e-9)), n_random=len(null), p=float(p)))
    print(f"{c} #{fid}: effect {eff:.4f} | random mean {null.mean():.4f} | p {p:.3f}")
abl = pd.DataFrame(abl); abl["p_bh"] = bh(abl.p.values); abl["causal"] = (abl.p_bh < 0.05) & (abl.effect > 0)
save_table(abl, "ablation_causal")
print(abl[["cls", "title", "feature", "n", "effect", "null_mean", "ratio_vs_random", "p", "p_bh", "causal"]].round(4).to_string(index=False))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3571, "status": "ok", "timestamp": 1791287503764, "user": {"displayName": "Simar Lamba", "userId": "09007720624843286617"}, "user_tz": -330} id="iZfiy5ic4FXn" outputId="a1a2afea-d6bf-4c43-ac18-7a091e33d65f"
pick = pd.concat([sample[sample.y_event == c].head(3) for c in top_classes]).reset_index(drop=True)
m1 = load_preds("M1", "event", "test").set_index("id").pred
pr = load_preds("saemax_L13", "event", "test").set_index("id").pred
examples = []
for i in range(0, len(pick), 8):
    chunk = pick.iloc[i:i + 8]
    cache, mask, ids = reader.run(chunk.narrative.tolist(), stop_at=13)
    f = (sae13.encode(cache[13].float()) * mask.unsqueeze(-1))[..., cand_t].cpu().numpy()
    for j, row in enumerate(chunk.itertuples()):
        n_t = int(mask[j].sum()) + 1
        toks = [dec(ids[j][t:t + 1]) for t in range(1, n_t)]
        acts = f[j, 1:n_t]
        top5 = np.argsort(-acts.max(0))[:5]
        examples.append(dict(id=row.id, true=f"{row.y_event} {titles[row.y_event].strip()}",
                             pred_m1=classes[int(m1[row.id])], pred_probe=classes[int(pr[row.id])],
                             preview=row.narrative[:70], tokens=toks,
                             acts={str(int(cand[k])): np.round(acts[:, k], 2).tolist() for k in top5 if acts[:, k].max() > 0}))
save_json(examples, WORK / "demo_examples.json")
print(len(examples), "demo examples saved")

# %% id="PJbFGTBTD6D5"
