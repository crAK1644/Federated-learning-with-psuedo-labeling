# Dawid-Skene için sonraki adımlar

Bu not öğretmene verilecek açıklama raporunun bir parçası değildir. Bundan sonra yapılacak teknik ve deneysel işleri ayrı tutmak için hazırlanmıştır.

## 1. Terminolojiyi sabitle

İlk olarak rapor, kod metrikleri ve sözlü açıklamalarda aşağıdaki kavramları birbirinden ayır:

- `rejected_count`: Hatalı, duplicate veya protokol doğrulamasından geçemeyen client mesajlarının sayısı.
- `excluded_clients`: Dawid-Skene fit'i için minimum annotation şartını karşılamayan client'lar. Mevcut active ayarda minimum değer 1 olduğu için bu, esas olarak hiçbir label vermeyen client anlamına gelir.
- `ds_applied`: Dawid-Skene sonucunun o round'da gerçekten broadcast edilip edilmediği.
- `fallback`: Tek tek client'ların değil, round-level Dawid-Skene sonucunun bırakılması ve majority vote kullanılması.
- `client weighting`: Bütün uygun client label'larının, öğrenilen confusion matrix'ler aracılığıyla sınıfa özel etkide bulunması.

Bu terminoloji üzerinde anlaşılmadan yeni bir weighting yöntemi tasarlamamak daha güvenli olur. Öğretmenin ``client feedback'' ile pseudo-label feedback'ini mi, yoksa client model update'ini mi kastettiği de netleştirilmeli. Mevcut Dawid-Skene yalnızca pseudo-label aggregation yapıyor.

## 2. Mevcut implementasyon için bağımsız sayısal doğrulama hazırla

Davranışsal reproduction tamamlandı; bir sonraki aşama aynı annotation matrix'i iki farklı implementasyona vermek olmalı.

1. Küçük ve sabit bir synthetic annotation matrix'i oluştur.
2. Aynı matrisi mevcut MAP-EM implementasyonuna ve `rater` referansına ver.
3. Aynı prior ve pseudocount'ları mümkün olduğu ölçüde eşleştir.
4. Aşağıdaki çıktıları karşılaştır:
   - Aggregate label accuracy
   - Posterior class olasılıkları
   - Client confusion matrix'lerinin yönü
   - Sınıf permutation'ı sonrası eşdeğerlik
   - Missing annotation davranışı
5. Bayesian Stan ve MAP-EM farkı nedeniyle birebir floating-point eşitliği değil, önceden tanımlanmış toleranslar ve davranışsal eşdeğerlik kullan.
6. Sonucu çalıştırılabilir bir script, sabit seed ve kısa bir tabloyla kaydet.

## 3. FL dışında üç kontrollü aggregation deneyi çalıştır

### Kolay yakınsama kontrolü

- 5 veya 7 güvenilir client kullan.
- Her client'ın accuracy'sini yaklaşık yüzde 90 yap.
- En az 1.000 ortak item üret.
- Beklenti: Dawid-Skene birkaç EM iterasyonunda yakınsamalı ve yüksek accuracy vermeli.

### Sistematik hata kontrolü

- Önce bütün client'ları güvenilir yap.
- Her aşamada bir güvenilir client'ı sistematik hatalı client ile değiştir.
- Majority ve Dawid-Skene accuracy eğrilerini birlikte çiz.
- Beklenti: Temiz durumda majority güçlü olabilir; sistematik hatalı client sayısı arttıkça Dawid-Skene göreli avantaj kazanmalı.

### Artificial tie kontrolü

- Anchor ve tie item'larını ayrı üret.
- Anchor item'lar client reliability'sini tanımlayacak bilgi taşımalı.
- Target item'larda 2-2 gibi bilinçli tie oluştur.
- Majority'nin tie-breaking doğruluğunu ve Dawid-Skene'in tie doğruluğunu ayrı raporla.
- Anchor verisi olmadan tamamen simetrik koşulu negatif kontrol olarak çalıştır. Beklenti: Modelin bunu güvenli biçimde tanımlanamaz olarak görmesi.

## 4. Gerçek veri için hedef sınıf çiftini seç

Sınıf çifti yalnızca ``en çok karıştırılan'' çift olduğu için seçilmemeli. Tamamen ayırt edilemez bir çift, Dawid-Skene için iyi bir pozitif kontrol olmaz.

Şu ana kadarki probe sonuçlarına göre:

- `gafgyt.tcp` ve `gafgyt.udp` preprocessing sonrasında neredeyse ayırt edilemez görünüyor. Bu çift pozitif kontrol yerine negatif kontrol olarak kullanılabilir veya preprocessing düzeltilmeden ana deneyde kullanılmamalı.
- `gafgyt.combo` ve `gafgyt.junk` doğrusal model için karıştırılabilir, fakat non-linear modellerle yüksek ölçüde ayrılabilir görünüyor. Bu nedenle ``zor fakat öğrenilebilir'' pozitif kontrol için daha uygun bir başlangıç adayı.

Seçimden önce her aday çift için şunları çıkar:

- Sealed test confusion matrix
- Linear probe accuracy
- Random forest ve 1-nearest-neighbor accuracy
- Sınıf başına örnek sayısı
- Client bazında sınıf dağılımı
- İlk round'lardaki pseudo-label confusion matrix

## 5. Kontrollü 50-round FL deneyini tasarla

Ana değişken yalnızca client specialization düzeni olmalı. Open set, test seti, model mimarisi, optimizer, learning rate, round sayısı ve client başına toplam private sample sayısı sabit tutulmalı.

İki veri dağılımı kullanılabilir:

1. **Dengeli dağılım:** Hedef iki sınıf client'lara mümkün olduğunca dengeli dağıtılır.
2. **Specialist dağılım:** Bazı client'lar birinci, bazıları ikinci hedef sınıfta uzmanlaşır. Toplam client örnek sayıları, arka plan sınıflarından örnek takası yapılarak sabit tutulur.

Her dağılım için en az şu aggregation kollarını çalıştır:

- Majority vote
- Dawid-Skene shadow: DS yalnızca ölçülür, majority broadcast edilir
- Dawid-Skene active: Kontrolleri geçerse DS, aksi hâlde majority broadcast edilir

Hybrid weighting üzerinde karar verilirse dördüncü kol olarak ekle. İlk pilotu tek seed ile çalıştır; pipeline doğrulandıktan sonra en az 5 seed kullan.

## 6. Hybrid weighting için tasarım kararı al

Mevcut sistem binary davranıyor: Ya DS tamamen kullanılıyor ya da majority'ye tamamen dönülüyor. Öğretmenin ikinci mesajı kademeli bir kullanım istiyorsa aşağıdaki seçeneklerden biri açıkça seçilmeli.

### Seçenek A: Mevcut güvenli fallback'i koru

- Fit güvenliyse Dawid-Skene posterior label'larını kullan.
- Fit güvenli değilse majority vote kullan.
- Avantajı: Güvenli ve kolay denetlenebilir.
- Dezavantajı: Reliability bilgisi 161 fallback round'unda tamamen bırakılıyor.

### Seçenek B: Scalar weighted vote

- Her client için confusion matrix diagonalinden bir reliability ağırlığı türet.
- Bütün ağırlıkları pozitif tut; hiçbir client'ı tamamen sıfırlama.
- Dezavantajı: Sınıfa özel uzmanlık ve sistematik hata yönü tek sayıya indirgenir; bu standart Dawid-Skene değildir.

### Seçenek C: Class-conditional weighted vote

- Her client'ın confusion matrix'ini doğrudan kullan.
- Posterioru majority prior ile shrink ederek label switching riskini azalt.
- DS güveni düştükçe posterioru kademeli olarak majority dağılımına yaklaştır.
- Bu seçenek öğretmenin ``herkesi kullan ama güvenilir olan daha etkili olsun'' talebine en yakın olabilir; fakat yeni bir yöntem olarak ayrıca tanımlanıp ablation ile test edilmelidir.

Herhangi bir hybrid yöntem geliştirilmeden önce kullanılacak formül, ağırlıkların hangi veriyle tahmin edildiği ve aynı round verisinin hem ağırlık öğrenmek hem sonuç ölçmek için kullanılıp kullanılmadığı yazılı hâle getirilmeli.

## 7. Her round için kaydedilecek metrikleri genişlet

- Global test accuracy
- Hedef iki sınıfın ayrı recall değerleri
- Hedef sınıf çifti için macro F1
- Sealed test confusion matrix
- Open-set pseudo-label accuracy
- Tie count ve tie item accuracy
- Vote margin dağılımı
- `ds_applied`
- `ds_status` ve ilk fallback nedeni
- EM iteration sayısı
- Majority-DS agreement
- Diagonal fraction ve reference diagonal fraction
- DS valid rate
- Client başına annotation coverage
- Client başına tahmin edilen confusion matrix

Ana performans ölçütü yalnızca son round accuracy'si olmamalı. Örneğin hedef iki sınıftan daha zayıf olanın recall değerinin 41-50. round ortalaması ana ölçüt olarak önceden belirlenebilir.

## 8. Deneyden önce başarı kriterlerini yaz

Sonuca baktıktan sonra eşik belirlemekten kaçınmak için aşağıdaki kriterleri önceden tanımla:

- Kolay synthetic senaryoda yakınsama ve minimum accuracy beklentisi
- Sistematik hata arttığında DS'nin majority'ye göre beklenen yönü
- Tie senaryosunda minimum iyileşme
- Active run'da kabul edilebilir minimum `ds_applied` oranı
- 50-round deneyinde ana metric ve seed aggregation yöntemi
- Hybrid yöntem kullanılırsa majority ve standard DS karşısındaki kabul kriteri

## 9. Kısa clarification toplantısında karara bağlanacak konular

1. ``Client weighting'' pseudo-label weighting mi, model update weighting mi?
2. Round-level safety fallback korunacak mı?
3. Hybrid weighting ayrı bir deney kolu olarak mı ele alınacak?
4. Pozitif kontrol için hangi sınıf çifti kullanılacak?
5. 50-round pilot kaç seed ile başlayacak?
6. Ana başarı metriği accuracy mi, hedef sınıf recall'u mu, macro F1 mı?
7. `rater` ile karşılaştırmada hangi çıktılar ve toleranslar yeterli kabul edilecek?

Bu kararlar alındıktan sonra uygulama sırası: bağımsız doğrulama, FL dışı kontrollü testler, tek seed 50-round pilot, kontrol ve ardından çoklu seed ana deney şeklinde olmalı.

---

## Aşama 2 sonucu (tamamlandı)

`scripts/validate_dawid_skene_reference.py` aynı sabit annotation matrisini hem bu repodaki
MAP-EM implementasyonuna hem de bağımsız bir referansa veriyor. Tek komut:

```bash
uv run --with crowd-kit python scripts/validate_dawid_skene_reference.py
```

**Referans seçimi:** Plan `rater`'ı adlandırıyor, fakat `rater` R + Stan (Bayesian). Deterministik
bir MAP-EM'i Bayesian bir örnekleyiciyle karşılaştırmak ikinci bir fark kaynağı ekler, daha temiz
bir karşılaştırma vermez. Bunun yerine aynı 1979 modelinin bağımsız Python implementasyonu olan
`crowd-kit` kullanıldı. `crowd-kit` bilerek proje bağımlılığı **değil** (transformers/tokenizers
zinciri çekiyor); `uv run --with` ile çalıştırılıyor, testi yoksa skip ediyor.

Deney tasarımı: 4 sınıf, 400 item, 7 client (4 güvenilir, 1 gürültülü, 1 sistematik hatalı
`2 -> 1`, 1 seyrek/abstain eden). Üç varyant: temel, eksik annotation, hiç annotate edilmemiş
item eklenmiş.

| Vaka | Bizim accuracy | crowd-kit | Majority | Label agreement | Ortalama posterior farkı |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.9700 | 0.9700 | 0.9625 | 1.0000 | 7.3e-07 |
| missing | 0.9575 | 0.9575 | 0.9325 | 1.0000 | 1.6e-06 |
| all_abstain | 0.9700 | 0.9700 | 0.9625 | 1.0000 | 7.3e-07 |

Ek doğrulamalar: sınıf permutation'ı her üç vakada identity, sistematik hatanın yönü iki
implementasyonda da `2 -> 1`, confusion matrix ortalama farkı 1e-06 mertebesinde. Determinism,
client sırasına karşı invariance ve "hiç annotate edilmemiş item'lar fit'i kaydırmıyor" kontrolleri
geçti.

Karşılaştırma arm'ı pseudocount'ları ~0'a çekiyor (crowd-kit regularize edilmemiş MLE fit ediyor)
ve permutation gate'lerini kapatıyor: gate'ler deployment safeguard'ı, algoritma özelliği değil.

**Bu iş sırasında bulunan bir hata:** hiç annotation almamış item'ın placeholder posterior'ı
M-step'te class prior sayımına giriyordu; all-abstain kolon eklemek fit'i kaydırıyordu. Düzeltildi
(`fix: exclude unobserved items from the Dawid-Skene class prior`).

`tests/protocol/test_dawid_skene_reference_validation.py` bunu teste bağlıyor ve ayrıca negatif
kontrol içeriyor: confusion matrix bilerek transpose edildiğinde karşılaştırmanın hata vermesi
gerekiyor. Doğrulayıcının kendisi başarısız olamıyorsa hiçbir şey kanıtlamaz.

---

## Aşama 3 sonucu (tamamlandı)

`scripts/controlled_aggregation_experiments.py` planın üç kontrollü deneyini tek komutta
çalıştırıyor. FL yok, model eğitimi yok, N-BaIoT yok: sadece aggregator, cevabı inşa gereği bilinen
veri üzerinde.

```bash
uv run python scripts/controlled_aggregation_experiments.py
```

Bütün beklentiler koda `EXPECTATIONS` olarak **çalıştırmadan önce** yazıldı; script beklenti
tutmazsa non-zero exit veriyor.

### 1. Kolay yakınsama

7 client, her biri yüzde 90 doğru, 1.000 ortak item, 20 replicate.

| Dawid-Skene | Majority | Medyan EM adımı | Fit pass rate |
| ---: | ---: | ---: | ---: |
| 0.9995 | 0.9995 | 4 | 1.00 |

Beklenti karşılandı: birkaç EM adımında yakınsıyor ve tavana oturuyor.

### 2. Sistematik hata sweep'i

7 client'ın içine, aynı yönde sistematik hata taşıyan client'lar teker teker konuyor (sınıf
2'de yüzde 85 oranında sınıf 1 diyorlar, diğer üç sınıfta yüzde 90 doğrular). Şekil:
`artifacts/validation/systematic_sweep.png`.

| Hatalı client | Majority | Dawid-Skene | Fark | Saldırılan sınıfın recall'u |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9995 | 0.9995 | -0.0000 | 0.9998 |
| 1 | 0.9981 | 0.9992 | +0.0011 | 0.9988 |
| 2 | 0.9898 | 0.9986 | +0.0087 | 0.9980 |
| 3 | 0.9400 | 0.9968 | +0.0568 | 0.9936 |
| 4 | 0.7901 | 0.9941 | +0.2040 | 0.9869 |
| 5 | 0.7603 | 0.9815 | +0.2212 | 0.9543 |

Beklenen yön çıktı: temiz havuzda fark yok, paylaşılan sistematik hata büyüdükçe Dawid-Skene
öne geçiyor.

**Dikkat edilmesi gereken nokta:** hatalı client'lar çoğunluğa geçtiğinde (4 ve 5) bile
Dawid-Skene çökmüyor. Bunun nedeni buradaki hatanın *sınıfa koşullu* olması: hatalı client dört
sınıfın üçünde hâlâ doğru, sadece birinde bozuk, ve confusion matrix bunu doğrudan temsil ediyor.
Bu, bu hata biçimine ait bir sonuç, genel bir garanti değil. Her yerde yanlış olan bir client veya
neredeyse bütün client'ların paylaştığı bir bias farklı bir rejim; bu deney onun hakkında kanıt
vermiyor. Bu yüzden çoğunluk bölgesinde hiçbir beklenti assert edilmiyor.

### 3. Yapay tie

Anchor item'lar hangi client'ın güvenilir olduğunu belirliyor; target item'lar iki güvenilir ve
iki güvenilmez client arasında birebir 2-2 bölünecek şekilde inşa ediliyor. Sadece tie item'ları
üzerindeki accuracy:

| Vaka | Anchor | Lowest-index tie-break | (replicate aralığı) | Rastgele tie-break | Dawid-Skene |
| --- | ---: | ---: | :---: | ---: | ---: |
| anchor'lı | 600 | 0.5050 | 0.17 - 0.79 | 0.5053 | **1.0000** |
| tanımlanamaz | 0 | 0.4710 | 0.23 - 0.77 | 0.4915 | 0.4578 |

İki sonuç:

1. Anchor verisi varken Dawid-Skene tie'ların tamamını doğru çözüyor, majority ise şansta.
2. Anchor olmadan iki hipotez birbirinin tam relabelling'i oluyor ve model bunu çözmüş gibi
   *görünmüyor* (0.4578, yani şans). Negatif kontrol çalışıyor. Bu önemli: buradaki skor yüksek
   çıksaydı, anchor'lı sonuç kanıt olmaktan çıkardı.

**`TIE_BREAK_PLAN.md` için ölçülmüş veri:** mevcut "en küçük sınıf indeksi kazanır" kuralının
tie doğruluğu replicate'ler arasında 0.17 ile 0.79 arasında geziniyor, ortalaması 0.5050. Yani
kural sinyale değil, sınıf indekslerinin geometrisine göre karar veriyor - tam olarak plan
notunun "arbitrary fiat, not a signal" iddiası, artık sayıyla.

`tests/protocol/test_controlled_aggregation_experiments.py` beklentileri teste bağlıyor, ayrıca
inşa edilen tie'ların gerçekten 2-2 olduğunu ve negatif kontrolün sessizce çözülebilir hâle
gelmediğini kontrol ediyor.

---

## Aşama 1 sonucu (tamamlandı)

Terminoloji tek yerde sabitlendi: `DAWID_SKENE_GLOSSARY.md`. Rapor, kod metrikleri ve sözlü
anlatım artık aynı tanımlara bakıyor.

Sözlükte dört bölüm var:

1. **Birbirine karışan dört "dışarıda kalma".** Bir client bir round'dan dört ayrı noktada
   düşebiliyor ve bunların karıştırılması bir run'ı yanlış okumanın en kolay yolu:
   `rejected_count` (mesaj doğrulamadan geçmedi, annotation matrix'e hiç girmedi),
   `ds_excluded_clients` (matrix'e girdi ama minimum annotation şartını karşılamadı, EM'den önce
   atıldı), `ds_eligible_clients` (fit'e giren), `all_abstain_count` (hiçbir client'ın label
   vermediği item).
2. **Fit / uygulama / fallback.** `ds_status` ilk hata nedenidir ve kodlar birbirini dışlar.
   `ds_applied` yalnızca fit bütün gate'leri geçtiğinde **ve** run active moddayken 1 olur -
   shadow modda kusursuz bir fit bile 0 raporlar, çünkü hiçbir şey broadcast edilmedi.
   `ds_applied` ile `ds_status == ok` aynı soru değil. Fallback round-level'dır: o round'un DS
   sonucu bırakılır ve majority broadcast edilir; client bazında veya kısmi fallback yoktur.
   Warm-up round'ları başarısızlık değildir, fallback oranına katılmamalıdır.
3. **Benzer görünen büyüklükler.** `valid_rate` ile `ds_valid_rate`, `ds_diagonal_fraction` ile
   `ds_reference_diagonal_fraction`, ve `ds_majority_agreement` ile `ds_disagreement_rate`.
   Sonuncu ikisi aynı mask üzerinde hesaplanıyor, yani birbirinin tümleyeni - bağımsız iki kanıt
   gibi raporlanmamalı.
4. **Kullanılmayacak kelimeler.** DS posterior'ı için "confidence" (client tarafındaki
   `confidence_*` metrikleri farklı bir büyüklük ve makalenin kendi terimi, onlar kalıyor), DS
   çıktısı için "consensus", excluded/rejected'ın birbirinin yerine kullanılması, ve hangisi
   olduğu söylenmeden "client feedback".

`client weighting` sözlükte açıkça **önerilmiş ama uygulanmamış** olarak işaretlendi: bugünkü
kodda hiçbir yol client ağırlıklandırmıyor, active path ya ağırlıksız majority ya da ağırlıksız
Dawid-Skene argmax. Bu, 9. bölümdeki 1. soruyla doğrudan bağlantılı.

Sözlüğün çürümemesi için `tests/unit/test_dawid_skene_glossary.py` yazıldı: kod bir metrik
yayınlayıp sözlük tanımlamıyorsa, ya da sözlük artık yayınlanmayan bir metriği tanımlıyorsa test
düşüyor. Status kod tablosu da `DS_STATUS_CODES` ile birebir karşılaştırılıyor. Test yazılır
yazılmaz eksik bir tanım buldu (`ds_reference_diagonal_fraction`).

Yan düzeltme: `MODEL_CARD.md` SSFL'i "broadcasts consensus hard labels" diye anlatıyordu; aggregator
artık seçilebilir olduğu için "aggregated hard labels (majority vote, or Dawid-Skene when enabled)"
oldu.

---

## Aşama 4 sonucu (tamamlandı)

`scripts/probe_class_pairs.py` gerçek N-BaIoT verisi üzerinde adayları ölçüyor. Bütün probe'lar
private split'te fit ediliyor, sealed test split'te değerlendiriliyor. Kriterler sonuçlara
bakılmadan önce `CRITERIA` olarak yazıldı.

```bash
uv run python scripts/probe_class_pairs.py
```

### Çok sınıflı probe'lar (sealed test)

| Probe | Accuracy |
| --- | ---: |
| Linear (logistic regression) | 0.7704 |
| Random forest | 0.8983 |
| 1-nearest-neighbor | 0.8851 |

### Adaylar

| Çift | Confusion mass | Linear | Forest | 1-NN | Rol |
| --- | ---: | ---: | ---: | ---: | --- |
| `gafgyt.tcp` / `gafgyt.udp` | 0.4992 | 0.4992 | 0.4992 | 0.4994 | negatif kontrol |
| `gafgyt.combo` / `gafgyt.junk` | 0.0017 | 0.6861 | 0.9983 | 0.9981 | pozitif kontrol |

Sıradaki en karışık çiftler (`benign`/`gafgyt.tcp`, `benign`/`gafgyt.scan`, ...) zaten 0.998
üzerinde ayrılıyor, yani gerçek bir aday yok.

**`gafgyt.combo` / `gafgyt.junk` dört ön-kayıtlı kriteri de geçiyor:** non-linear ayrılabilir
(0.9983 >= 0.95), linear zorlanıyor (0.6861 <= 0.90), client coverage yeterli (16 ve 9 client,
>= 3), open-set desteği yeterli (900 ve 900, >= 200). "Zor ama öğrenilebilir" tanımına tam
oturuyor: aggregation'ın fark yaratabileceği boşluk burada.

### `gafgyt.tcp` / `gafgyt.udp`: ayırt edilemez değil, yok edilmiş

Bu çift negatif kontrol kriterini geçiyor (forest 0.4992 <= 0.80) ama nedeni önemli, çünkü
"bu iki saldırı birbirine benziyor" ile "pipeline farkı çöpe attı" aynı şey değil:

| Sınıf | Sealed test'te farklı satır | Ortalama feature std |
| --- | ---: | ---: |
| `gafgyt.tcp` | 1.800 örnekte **5** | 0.002888 |
| `gafgyt.udp` | 1.800 örnekte **1** | 0.000001 |
| diğer dokuz sınıf | 1.400-1.800 arası (neredeyse hepsi farklı) | 0.009 - 0.094 |

Yani `gafgyt.udp`'nin bütün test örnekleri tek bir vektör, ve `gafgyt.udp` satırlarının tamamı
bir `gafgyt.tcp` satırıyla byte-byte aynı. Hiçbir model, hiçbir aggregator bunu ayıramaz.

Kaybın nerede olduğu izlendi:

1. **Sampling değil.** Audit trail'de bu 1.800 test satırı 1.800 farklı kaynak satırından geliyor.
2. **Ham veri değil.** Ham CSV'de (Ecobee, ilk 2.000 satır) `tcp` 1.847, `udp` 1.892 farklı satır
   içeriyor.
3. **float32 cast.** Aynı ham satırlar min-max scaler'dan geçirildiğinde float64'te 1.847 farklı
   satır kalıyor, float32'ye çevrildiğinde **7**'ye düşüyor. `combo` aynı işlemde 2.000/2.000
   kalıyor.

Mekanizma: global min-max aralığı ~5.7e17 mertebesindeki outlier'lar tarafından belirleniyor;
`tcp`/`udp`'nin sınıf içi değişimi bu aralığa bölündükten sonra float32 çözünürlüğünün altına
düşüyor. float64 bunu koruyor, float32 yuvarlayıp atıyor.

**Öneri:** çift negatif kontrol olarak kalsın, ama preprocessing ana deneyden önce
değiştirilmesin - makalenin kendisi min-max kullanıyor ve scaler'ı değiştirmek bütün
reprodüksiyonu kaydırır. Bu bulgu bir deviation notu olarak kaydedilmeli, sessizce düzeltilecek
bir bug olarak değil.

### Gerçek run'da ne oluyor

Kayıtlı 50-round shadow run'ının aggregation audit'i, open-set ground truth'a karşı skorlandı
(ground truth yalnızca offline, data-prep audit trail'inden okunuyor; server onu hiç görmüyor):

| Round | Majority pseudo-label accuracy | Valid rate |
| ---: | ---: | ---: |
| 1 | 0.1969 | 0.9627 |
| 10 | 0.6989 | 0.9998 |
| 25 | 0.7078 | 0.9994 |
| 50 | 0.7361 | 0.9993 |

Çift bazında:

- `tcp`/`udp`: round 1'de bütün `udp` örnekleri `tcp` etiketlendi (898/898); round 10'dan sonra
  tam tersi, bütün `tcp` örnekleri `udp` etiketlendi (899/899) ve `udp` tamamen doğru göründü.
  Yani hangi ismin kazandığına yazı-tura karar veriyor. Bu, yukarıdaki collapse'in FL tarafındaki
  görüntüsü.
- `combo`/`junk`: round 10'da `junk -> combo` 800, ters yön 35; round 50'de 764 ve 16. `combo`
  büyük ölçüde doğru (883/899), `junk` büyük ölçüde yanlış (132/896). Yani **asimetrik, sınıfa
  koşullu** bir hata - Dawid-Skene'in confusion matrix'inin doğrudan temsil ettiği hata biçimi,
  ve Aşama 3'teki sistematik hata deneyinin gerçek veri karşılığı.

### 9. bölümdeki 4. soruya cevap

Pozitif kontrol `gafgyt.combo` / `gafgyt.junk`, negatif kontrol `gafgyt.tcp` / `gafgyt.udp`.
Öğretmene sorulacak olan artık "hangi çift" değil, "bu kanıtla bu seçim uygun mu".

---

## Aşama 7 sonucu (tamamlandı)

Listedeki on beş kalemin çoğu zaten kaydediliyordu; eksik olan iki şey vardı. Birincisi, sealed
open-set etiketlerine karşı ölçülen her şey (pseudo-label accuracy, tie item accuracy, hedef çift
dilimleri) - bunlar sunucuda hesaplanamaz, çünkü sunucu open-set ground truth'u hiç görmemeli.
İkincisi, client başına iki büyüklük.

### Sunucu tarafına eklenen (`src/ssfl/strategies/ssfl.py`)

- `ds_coverage_min` / `ds_coverage_mean` / `ds_coverage_max` - her client'ın open-set'in ne kadarını
  gerçekten etiketlediği. `participating_*` ile aynı eksen değil: o, item başına client sayar; bu,
  client başına item sayar. Warm-up round'larında da yazılıyor ve annotation matrisindeki bütün
  client'ları kapsıyor, sonradan exclude edilenler dahil. `DAWID_SKENE_GLOSSARY.md`'ye eklendi,
  yani rot guard testi bu anahtarları da koruyor.
- Client başına tahmin edilen confusion matrix, audit `.npz` dosyasına
  `dawid_skene_confusion` + `dawid_skene_confusion_clients` olarak. **Annotation matrisiyle aynı
  restricted gate'in arkasında** (`dawid_skene_save_annotations`, varsayılan kapalı, round bazında
  opt-in) - per-client davranış, annotation matrisiyle aynı hassasiyette. Satır sırası matristen
  geri kazanılamadığı için eligible sender listesi yanında gidiyor
  (`DawidSkeneFit.eligible_senders`).

### Offline ledger (`scripts/round_metrics.py`)

```bash
uv run python -m scripts.round_metrics artifacts/runs/<run_id>
```

Sealed etiketleri `scripts/ds_headroom.py`'daki mevcut okuyucudan alıyor (yeni bir etiket dosyası
yazılmıyor - `artifacts/data`'ya dosya eklemek manifest hash'ini, dolayısıyla bütün `run_id`'leri
değiştirir). Round başına ürettikleri:

| Sütun | İçerik |
| --- | --- |
| `broadcast_accuracy` | O round gerçekten yayınlanan etiketlerin open-set doğruluğu |
| `majority_accuracy`, `dawid_skene_accuracy` | Shadow run'da iki kolu ayrı ayrı, aynı satırda |
| `tie_count`, `tie_rate`, `tie_accuracy` | Tie item'lar ve **yalnız o item'lardaki** doğruluk |
| `majority_tie_accuracy`, `dawid_skene_tie_accuracy` | Aynı tie maskesi, iki aggregator |
| `margin_p10`, `margin_median`, `margin_mean` | Vote margin dağılımı |
| `pair_a_recall`, `pair_b_recall`, `pair_a_precision`, `pair_b_precision`, `pair_macro_f1` | Hedef çift, open-set üzerinde |
| `pair_confusion_ab`, `pair_confusion_ba` | Çiftin iki yönü ayrı - asimetri tek sayıda kaybolur |
| `test_accuracy`, `test_pair_*_recall`, `test_pair_*_f1` | Sunucunun sealed test'te zaten yazdıkları |
| `ds_*` | `metrics.parquet`'ten olduğu gibi |

Run seviyesinde ayrıca `first_fallback_round` + `first_fallback_status` (6. round'da düşen run ile
44. round'da düşen run aynı ortalamayı verir ama aynı şey değildir) ve ana metrik adayı
`weaker_recall_last10_mean` - planın önerdiği "zayıf sınıfın 41-50. round recall ortalaması".

Abstain edilen item'lar **yanlış sayılmıyor**, maskeden çıkarılıyor: yayınlanmayan bir item ile
yanlış yayınlanan bir item aynı hata değil. Tie'ı olmayan bir round'un tie accuracy'si `NaN`,
`0.0` değil - aksi halde ortalamalar aşağı çekilir. İkisi de testle sabitlendi
(`tests/unit/test_round_metrics.py`).

### Kayıtlı shadow run'da ilk çıktı

| Round | broadcast | majority | dawid_skene | tie count | majority tie acc | DS tie acc | junk recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1969 | 0.1969 | - | 1276 | 0.1019 | - | 0.0000 |
| 10 | 0.6989 | 0.6989 | - | 362 | 0.2818 | - | 0.1067 |
| 25 | 0.7078 | 0.7078 | 0.5548 | 190 | 0.2895 | 0.2000 | 0.1178 |
| 50 | 0.7361 | 0.7361 | 0.5305 | 230 | 0.3783 | 0.2217 | 0.1467 |

Shadow kontratı görünüyor: `broadcast` ile `majority` her round'da birebir aynı. Yeni bilgi tie
sütunlarında - Dawid-Skene'in asıl iddiası tam olarak tie item'larda majority'yi geçmesiydi;
kayıtlı run'da **tie'larda da geride** (0.2000 vs 0.2895, 0.2217 vs 0.3783). REPRODUCIBILITY #33
ve #37'deki no-go kararıyla tutarlı, ve o kararı bu sefer tie ekseninde ölçüyor.

`pair_macro_f1` 0.43 - 0.47 arasında ve `pair_confusion_ba` 800'den 764'e çok yavaş düşüyor:
`gafgyt.junk` 50 round boyunca büyük ölçüde `gafgyt.combo` olarak etiketleniyor. Ana deneyin
hareket ettirmesi gereken sayı bu.


## Aşama 5 sonucu (tamamlandı - tasarım ve konfigürasyon)

Deney tasarımı ve altı kolun konfigürasyonu yazıldı; koşum GPU'ya bağlı ve bu ortamda
çalıştırılmadı. Aşağıdaki her şey "hazır ve doğrulanmış konfigürasyon" seviyesindedir, sonuç
değildir.

### Eksik olan tek yetenek: scenario 4

Aşama 5'in şartı net: *"client başına toplam private sample sayısı sabit tutulmalı"*. Mevcut üç
senaryo bunu veremiyor - scenario 1 ve 2 shard tabanlı, scenario 3 Dirichlet ve client toplamlarını
kendisi değiştiriyor. Yani mevcut senaryolarla "specialist" bir kol kurulsa, DS ile majority
arasındaki her fark aynı zamanda bir veri hacmi farkı olurdu.

Bunun için `src/ssfl/data/partition.py` içine **scenario 4** eklendi:

- Her client tam olarak `private_count` satır alır, specialization seviyesi ne olursa olsun. Hedef
  çiftten alınan/verilen satırlar arka plan sınıflarından takas edilerek dengelenir.
- `specialization` **tek sürekli bir knob**: `0.0` düz dağıtım, `1.0` hedef sınıfı tamamen kendi
  uzmanlarına verir. Dengeli ve specialist kollar aynı kod yolunu kullanır - iki ayrı partition
  algoritması değil.
- Specialist'ler ardışık değil **strided** seçilir (`arange(rank, num_clients, len(present))`), ki
  index ile korelasyonlu başka bir etki yan yolcu olarak binmesin.
- Arka plan sınıfları tek bir karıştırılmış havuzdan dağıtılır: arka plan kompozisyonu sinyal
  değil, gürültü kalır.
- Tamsayı bölüşümü `_largest_remainder` ile yapılır, böylece tahsisler tam olarak toplama eşit
  çıkar ve tie'lar deterministik olarak düşük index'e gider.

Scenario 4 **opt-in**: yalnızca `--target-classes` verilirse üretilir. Sebep manifest hash'i -
varsayılan hazırlanmış veri kümesi, hash'i ve dolayısıyla **mevcut her `run_id`** scenario 4
eklenmeden önceki ile birebir aynı kalır. `run_kind=extension` config seviyesinde zorunlu tutulur,
çünkü bu bir paper senaryosu değil.

Dokuz test bunu sabitliyor (`tests/unit/test_partition_scenario_4.py`): her specialization'da eşit
satır sayısı, her private satırın tam bir kez kullanılması, specialization'ın **yalnızca** hedef
çifti yoğunlaştırması (arka plan sahipleri 1'den fazla düşemez), tam specialist'in çiftin tek bir
sınıfını tutması, determinizm ve seed bağımlılığı, hedef çifti içermeyen cihazın da temiz
bölünmesi, `[0,1]` dışındaki specialization'ın reddedilmesi.

### Sabit tutulanlar ve tek değişken

| Sabit | Değişken |
| --- | --- |
| Open set, test seti, scaler, split'ler (aynı seed, aynı prepare) | Client specialization düzeni (`data_path`) |
| Backbone (cnn), optimizer, learning rate (1e-4), batch size (80) | Aggregation kolu (`ssfl_hard_aggregation`) |
| Round sayısı (50), local epoch (5), seed (2023) | |
| **Client başına private satır sayısı** (scenario 4 garantisi) | |

### Altı kol

`configs/controlled_pair.yaml` (temel profil) + `configs/experiments_controlled_pair.yaml` (matris).
Doğrulandı: altı giriş de `build_matrix_configs` ile hatasız çözülüyor.

| Kol | data_path | ssfl_hard_aggregation |
| --- | --- | --- |
| `controlled_specialist_majority` | `artifacts/data-specialist` | `majority` |
| `controlled_specialist_shadow` | `artifacts/data-specialist` | `dawid_skene_shadow` |
| `controlled_specialist_active` | `artifacts/data-specialist` | `dawid_skene` |
| `controlled_balanced_majority` | `artifacts/data-balanced` | `majority` |
| `controlled_balanced_shadow` | `artifacts/data-balanced` | `dawid_skene_shadow` |
| `controlled_balanced_active` | `artifacts/data-balanced` | `dawid_skene` |

Bir dağılım içinde üç giriş **tek bir anahtarda** farklılaşıyor; dağılımlar arasında majority
girişleri **tek bir anahtarda** farklılaşıyor. Aradaki fark başka hiçbir şeyle açıklanamaz.

Hedef çift aşama 4'ten: `gafgyt.combo` (1) / `gafgyt.junk` (2). `gafgyt.tcp`/`gafgyt.udp` negatif
kontroldür ve burada **kullanılmamalı** - float32 collapse yüzünden öğrenilemez olması
aggregation ile ilgisi olmayan bir sebep.

### Koşum sırası ve bütçe

Veri kökleri (aynı seed, aynı split'ler, yalnızca client assignment manifest'i farklı):

```bash
uv run python -m ssfl.data.prepare_data --input data --output artifacts/data-balanced \
    --seed 2023 --target-classes 1 2 --target-specialization 0.0
uv run python -m ssfl.data.prepare_data --input data --output artifacts/data-specialist \
    --seed 2023 --target-classes 1 2 --target-specialization 1.0
uv run python -m ssfl.experiments.run_suite --matrix configs/experiments_controlled_pair.yaml
```

Matris **specialist bloğu ile başlıyor**: DS'in yardım etmesi beklenen rejim orası
(REPRODUCIBILITY #35(b) izole ölçümde +0.29 - +0.40), dolayısıyla negatif sonucun en bilgilendirici
olduğu yer de orası. Blok içinde majority ilk, çünkü diğer ikisi ona göre okunuyor ve tek başına
anlamlı olan tek kol o.

Ölçülen ~144 sn/round bu donanımda kol başına ~2 saat, tüm matris ~12 saat. GPU dolu, kollar
paralelleştirilemez. Pilot **tek seed** - amacı etkiyi ölçmek değil, pipeline ve partition'ı
doğrulamak; n=1 ile hiçbir fark kanıt değil ve #34 gürültü tabanı zaten yarım puan civarı. Beş seed
ancak bu matris temiz koştuktan ve aşama 8 kapıları geçtikten sonra.

## Aşama 6 sonucu (karar verildi)

**Karar: Seçenek B (scalar weighted vote), majority'ye doğru shrink edilmiş hâlde - ve yalnızca
aşama 8'in birincil kapısı geçerse inşa edilecek.**

### Formül

Eligible client `j` için DS'in kendi confusion matrix'inden türetilen balanced accuracy:

```
â_j = (1/K) * Σ_c M_j[c, c]          M_j[c, k] = P(client j "k" der | gerçek c)
ā   = eligible client'lar üzerinde â_j ortalaması
w_j(λ) = 1 + λ * (â_j - ā)           λ ∈ [0, 1]
```

Item `i` ve sınıf `k` için skor, ağırlıklı oy:

```
s(i, k) = Σ_{j : a_ji = k} w_j(λ)
global_label(i) = argmin over argmax_k s(i, k)      (tie'lar hâlâ en düşük sınıf index'i)
```

`λ = 0` her ağırlığı 1 yapar ve **bit bazında mevcut majority vote'u** üretir. Shrink budur:
tek bir dial, ve sıfır ucu mevcut davranışın kendisi. `â_j ∈ [0,1]` olduğundan `w_j ∈ [0,2]`;
hiçbir client sıfırlanmaz, öğretmenin "herkesi kullan ama güvenilir olan daha etkili olsun"
talebi bu.

ABSTAIN oy vermez - eksik veri, sınıf değil. Bu noktada mevcut `aggregate_votes` ile aynı.

### Neden B, C değil

C'nin sınıfa özel ağırlıkları client başına `K² = 121` parametre ister. Bu ölçekte ~25 eligible
client ve 900 open-set örneği var: ~22.500 annotation'dan ~3.025 hücre tahmin edilecek, çoğu hücre
neredeyse boş. B ise client başına **tek sayı**, 25 parametre. C ayrıca aşama 3'ün ölçtüğü tie
avantajını korumaz - orada avantajı yaratan anchor'lardı, ağırlık çözünürlüğü değil.

A ise reddedildi çünkü sorunun kendisi o: 161 fallback round'unda reliability bilgisi tamamen
atılıyor. Ama A **fallback olarak kalıyor** - `λ` shrink'i güvenli olmayan bir fit'i güvenli
yapmaz, sadece güvenli bir fit'in etkisini sınırlar. Permutation ve agreement kapıları aynen
yerinde kalır; ağırlıklar yalnızca kapılardan geçmiş bir fit'ten türetilir.

### Ağırlıklar hangi veriyle tahmin ediliyor - ve aynı veri mi ölçüyor?

**Evet, aynı round'un verisi hem ağırlıkları öğreniyor hem onlarla etiketleniyor. Bu protokolde
kaçışı yok** ve açıkça yazılması gereken şey buydu: annotate edilmiş tek veri open set, ve
tutulmuş (held-out) annotation yok. Üç şey bunu sınırlıyor:

1. `λ` shrink'i, bu kendine referanslı tahminin bir etiketi ne kadar oynatabileceğini üstten
   sınırlar. `λ = 0`'da hiç oynatamaz.
2. Ağırlıklar yalnızca permutation/agreement kapılarından geçmiş fit'lerden gelir; kapıların en
   önemlisi (`majority_agreement`) fit'in **dışında** bir çıpaya bağlı.
3. **Değerlendirme double-dip değil**: başarı, estimator'ın hiç görmediği mühürlü open-set ve test
   etiketlerine karşı offline ölçülüyor (`scripts/round_metrics.py`).

Bir sonraki round'un ağırlıklarını bir önceki round'dan almak (warm start benzeri) düşünüldü ve
alınmadı: pseudo-label kalitesi round'lar arası hızla değişiyor, bayat bir ağırlık kendi
double-dipping'inden daha kötü bir bias getirir ve `dawid_skene_warm_start` zaten `false`.

### Ne zaman inşa edilecek

Hiçbir satır kod yazılmadı. Hybrid ancak aşama 8'in `primary_min_ds_tie_advantage` kriteri **ve**
bütün validity kapıları geçerse inşa edilecek. Gerekçe: tie'lar ağırlıkların argmax'ı
değiştirebileceği **tek** yer; DS orada majority'nin keyfi tie kuralını yenemiyorsa hiçbir `λ`
onu kurtaramaz. Kayıtlı shadow run zaten tersini ölçüyor (aşama 7: 0.2217 vs 0.3783), yani şu
anki kanıt hybrid'e karşı.
