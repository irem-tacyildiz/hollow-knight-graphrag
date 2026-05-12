#  Hollow Knight Universe — GraphRAG

Hollow Knight ve Silksong oyun wiki verisi üzerinde **Graph-based Retrieval-Augmented Generation (GraphRAG)** sistemi. Knowledge Graph, Graph Attention Network (GAT) ve dual-retriever RAG pipeline birleştirilerek oyun evreni hakkındaki karmaşık sorulara detaylı ve kaynak destekli cevaplar üretilmektedir.

---

##  Mimari

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│   Wiki Verisi   │────▶│  Knowledge Graph  │────▶│   GNN (GAT)    │
│  (1674 sayfa)   │     │  (1674 düğüm,    │     │  Link Pred.    │
│  MediaWiki API  │     │   160K+ kenar)   │     │  AUC: 0.96     │
└─────────────────┘     └────────┬─────────┘     └───────┬────────┘
                                 │                       │
                                 │                 Node Embeddings
                                 ▼                       │
                        ┌──────────────────┐             │
      Kullanıcı ──────▶│  Dual Retriever   │◀────────────┘
      Sorusu           │                    │
                       │ ┌──────────────┐  │    ┌──────────────┐
                       │ │ 1. Entity    │  │    │              │
                       │ │    Arama     │  │───▶│   Gemma 4    │───▶ Cevap
                       │ ├──────────────┤  │    │   E4B-it     │
                       │ │ 2. Vektör    │  │    │   (LLM)      │
                       │ │    Arama     │  │    │              │
                       │ ├──────────────┤  │    └──────────────┘
                       │ │ 3. Graf      │  │
                       │ │    Arama     │  │
                       │ └──────────────┘  │
                       └──────────────────┘
```

---

##  Notebook İçeriği

Notebook 8 ana bölüm ve 35 hücreden oluşmaktadır:

### Bölüm 1 — Kurulum (Cell 3-4)
Gerekli kütüphanelerin yüklenmesi ve import edilmesi.

### Bölüm 2 — Wiki'den Veri Çekme (Cell 6-8)
- MediaWiki API (`/mw/api.php`) ile tüm sayfa başlıklarının listelenmesi (1674 sayfa)
- Her sayfanın içeriğinin çekilmesi — `BeautifulSoup` ile HTML'den temiz metin, dahili linkler, kategoriler ve tablolar ayrıştırılır
- Google Drive'a kaydetme (daha önce çekilen sayfalar tekrar çekilmez)
- Veri kalitesi kontrolü

### Bölüm 3 — Entity & Relation Çıkarma (Cell 10-12)
- **Entity tiplendirme** — 3 katmanlı sistem: manuel etiketleme → kategori bazlı → metin bazlı. 10 tip: enemy, item, npc, boss, area, skill, lore, quest, mechanic, other
- **Relation çıkarma** — İki yöntem:
  - Link-based: Wiki dahili linklerinden REFERENCES ilişkisi (~159K kenar)
  - Pattern-based: Regex ile LOCATED_IN, DROPS, REQUIRES, UNLOCKS, GUARDS, SELLS, GIVES, DEFEATED_BY, PART_OF ilişkileri

### Bölüm 4 — Knowledge Graph (Cell 14-16)
- `NetworkX` ile yönlü graf oluşturma (1674 düğüm, 160K+ kenar)
- PageRank ile en merkezi düğümlerin hesaplanması
- `matplotlib` ile entity/relation tip dağılımı grafikleri
- `PyVis` ile interaktif HTML graf görselleştirmesi

### Bölüm 5 — GNN: Graph Attention Network (Cell 18-23)
- Grafın `PyTorch Geometric` formatına dönüştürülmesi (node feature: entity tipinin one-hot encoding'i)
- `RandomLinkSplit` ile %80/%10/%10 train/val/test bölünmesi + negatif örnekleme
- GAT modeli: 2 katmanlı GATConv (4 attention head → 1 head), dropout 0.3, 32 boyutlu embedding
- 200 epoch eğitim (Adam, BCEWithLogitsLoss)
- Eğitim grafikleri (loss, AUC, AP)
- Node embedding çıkarma + t-SNE görselleştirme

### Bölüm 6 — GraphRAG Pipeline (Cell 25-31)
- Metin chunk'lama (500 kelime, 100 overlap) + `ChromaDB`'ye embedding yükleme (`all-MiniLM-L6-v2`)
- Graf arama fonksiyonları (BFS komşuluk, en kısa yol)
- `Gemma 4 E4B-it` LLM yükleme (`bfloat16`)
- **İyileştirme 1:** Entity Classification — "other" düğümlerin akıllı yeniden sınıflandırılması
- **İyileştirme 2:** Semantic Relations — CHILD_OF, SIBLING_OF, CREATED_BY, BOSS_OF gibi 20 yeni ilişki kalıbı
- **Ana pipeline:** `graphrag_answer()` — entity eşleşme (uzun isimler öncelikli) + vektör arama + graf arama + source sayfa ekleme + LLM cevap üretme
- Test soruları

### Kaydet / Geri Yükle (Cell 32-33)
- Google Drive'a checkpoint kaydetme
- Bağlantı kesildikten sonra tek hücreyle geri yükleme (3-4 dk)

### Demo — Chat Arayüzü (Cell 34)
- `Gradio` ile Hollow Knight temalı interaktif chat arayüzü
- `share=True` ile dış link

---

##  Kullanılan Teknolojiler

| Bileşen | Teknoloji | Amaç |
|---------|-----------|------|
| Veri toplama | `requests`, `BeautifulSoup` | MediaWiki API ile scraping |
| Knowledge Graph | `NetworkX` | Graf oluşturma, PageRank, BFS |
| GNN | `PyTorch Geometric` (GATConv) | Node embedding, link prediction |
| Vektör DB | `ChromaDB`, `sentence-transformers` | Semantik metin araması |
| LLM | `Gemma 4 E4B-it` (Google) | Cevap üretme |
| Görselleştirme | `PyVis`, `matplotlib`, `t-SNE` | Graf ve embedding görselleştirme |
| Arayüz | `Gradio` | İnteraktif chat demo |
| Depolama | Google Drive | Checkpoint ve veri kalıcılığı |

---

##  Nasıl Çalıştırılır?

1. Notebook'u **Google Colab**'a yükle
2. `Runtime → Change runtime type → GPU (A100 veya L4)` seç
3. **HuggingFace Token:**
   - [huggingface.co](https://huggingface.co) hesabı aç
   - [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) sayfasında lisansı kabul et
   - Settings → Access Tokens → token oluştur
   - Colab'da sol menü →  Secrets → `HF_TOKEN` adıyla kaydet
4. `Runtime → Run all` ile tüm hücreleri çalıştır
5. Gradio arayüzünden veya `graphrag_answer("sorunuz")` ile soru sor

**Bağlantı kesildikten sonra:** KAYDET hücresi çalıştırılmışsa, sadece GERİ YÜKLE hücresini çalıştırmak yeterlidir.

---

##  Lisans

MIT License

##  Yasal Uyarı (Disclaimer)

Bu proje tamamen eğitim, araştırma ve yapay zeka modeli geliştirme amaçlı oluşturulmuş, ticari olmayan **resmi dışı** bir hayran projesidir. Hollow Knight, Hollow Knight: Silksong ve ilgili tüm oyun içi materyallerin, karakterlerin ve evren tasarımının fikri mülkiyet ve telif hakları **Team Cherry**'ye aittir.

Projede kullanılan bilgi ağı (Knowledge Graph) verileri, topluluk tarafından oluşturulan wiki sayfalarından çekilmiştir. Bu GraphRAG sisteminin Team Cherry ile hiçbir resmi bağı bulunmamaktadır. 