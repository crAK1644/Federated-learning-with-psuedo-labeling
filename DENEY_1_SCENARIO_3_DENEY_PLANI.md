# Deney 1 — Scenario 3 üzerinde MV, DS-only ve Hybrid karşılaştırması

## 1. Deneyin amacı

Bu deneyin temel araştırma sorusu şudur:

> Aynı federated learning koşullarında, client'lardan gelen pseudo-label'ları Dawid–Skene ile birleştirmek Majority Vote'a göre daha iyi bir global model oluşturuyor mu?

Üç yöntem karşılaştırılacaktır:

1. **MV-only:** Her round yalnızca Majority Vote kullanılacaktır.
2. **DS-only:** Her round yalnızca Dawid–Skene sonucu kullanılacaktır. Majority Vote'a fallback yapılmayacaktır.
3. **Hybrid:** Dawid–Skene güvenlik kontrollerini geçtiğinde DS, geçmediğinde Majority Vote kullanılacaktır.

Her kol 200 communication round çalışacaktır. Toplam deney bütçesi:

$$
3 \times 200 = 600 \text{ communication round}
$$

Bu üç koşul arasında aggregation yöntemi dışında hiçbir şey değiştirilmeyecektir.

## 2. Kullanılacak veri dağılımı: Scenario 3

Deney, mevcut **Scenario 3** veri bölüşümü üzerinde yapılacaktır.

| Özellik | Değer |
|---|---:|
| Client sayısı | 89 |
| Toplam private training örneği | 62.300 |
| Server ile client'lar arasında ortak kullanılan open set | 8.900 |
| Test seti | 17.800 |
| Sınıf sayısı | 11 |
| Partition yöntemi | Dirichlet |
| Dirichlet $\alpha$ | 0.1 |
| Seed | 2023 |
| Bir client'taki minimum örnek | 7 |
| Bir client'taki maksimum örnek | 1.581 |
| Bir client'taki ortalama örnek | 700 |
| Bir client'taki minimum sınıf sayısı | 1 |
| Bir client'taki maksimum sınıf sayısı | 8 |

Sınıflar:

| ID | Sınıf |
|---:|---|
| 0 | benign |
| 1 | gafgyt.combo |
| 2 | gafgyt.junk |
| 3 | gafgyt.scan |
| 4 | gafgyt.tcp |
| 5 | gafgyt.udp |
| 6 | mirai.ack |
| 7 | mirai.scan |
| 8 | mirai.syn |
| 9 | mirai.udp |
| 10 | mirai.udpplain |

Scenario 3 IID değildir. Client'ların örnek sayıları ve sahip oldukları sınıflar ciddi biçimde farklıdır. Özellikle fiziksel Device 3 ve Device 7 üzerinde Mirai sınıfları bulunmamaktadır.

Bu durum deney için önemlidir çünkü Majority Vote bütün client'ları eşit kabul ederken Dawid–Skene, client'ların sınıf bazındaki davranışlarını öğrenmeye çalışır.

Kullanılacak dataset manifest hash'i:

```text
54023e3de197d9dc16f50a89425d5c44bb594486210aacf16ecaa8cccc86ba65
```

Üç run başlamadan önce bu hash'in aynı olduğu otomatik olarak doğrulanmalıdır.

## 3. Değişmeyecek ortak deney ayarları

Üç deney kolunda aşağıdaki ayarlar birebir aynı olacaktır:

| Parametre | Değer |
|---|---:|
| Algorithm | SSFL |
| Backbone | CNN |
| Scenario | 3 |
| Communication round | 200 |
| Seed | 2023 |
| Local epoch | 5 |
| Batch size | 80 |
| Learning rate | $10^{-4}$ |
| Threshold policy | Median |
| Discriminator | Enabled |
| Voting mode | Enabled |
| Label representation | Hard |
| Device | CUDA |
| Deterministic execution | True |
| Client GPU allocation | 0.125 |
| Aynı anda çalışan client sayısı | 8 |
| Warm-up round | 0 |
| Client participation | Her round 89 client |

Her run:

- Aynı başlangıç modelinden başlayacaktır.
- Diğer run'ın checkpoint'inden devam etmeyecektir.
- Aynı client ID'lerini kullanacaktır.
- Aynı private data partition'ını kullanacaktır.
- Aynı 8.900 open sample üzerinde pseudo-label üretecektir.
- Aynı test setiyle değerlendirilecektir.
- Aynı kod commit'iyle çalıştırılacaktır.
- Aynı donanım ve yazılım ortamında çalıştırılacaktır.

Bir run başladıktan sonra kod veya config değiştirilirse karşılaştırma geçerliliğini kaybeder. Böyle bir durumda üç run da yeni kod sürümüyle baştan çalıştırılmalıdır.

## 4. Üç deney kolunun kesin tanımı

### 4.1. MV-only

Her open sample için client'ların geçerli tahminleri sayılacaktır:

$$
V_{i,c} = \sum_{j=1}^{J} \mathbf{1}(y_{ij}=c)
$$

Global pseudo-label:

$$
\hat{y}^{MV}_i = \arg\max_c V_{i,c}
$$

Eşitlik varsa mevcut deterministik kural kullanılacaktır: en küçük class index seçilecektir.

Server client'lara şunları broadcast edecektir:

- Majority Vote global pseudo-label'ları
- Common valid mask

Bu kolda:

- Online aggregation yolunda Dawid–Skene çalıştırılmayacaktır.
- DS sonucu training kararlarını etkilemeyecektir.
- DS fallback diye bir durum olmayacaktır.
- Client güvenilirlikleri istenirse kaydedilen annotation'lar üzerinden run sonrasında, offline olarak hesaplanacaktır.

Önerilen config değeri:

```yaml
ssfl_hard_aggregation: majority
```

### 4.2. DS-only

Client annotation matrisi:

$$
A \in \{-1,0,\ldots,10\}^{89 \times 8900}
$$

Burada:

- $A_{j,i}$: client $j$'nin sample $i$ için verdiği pseudo-label
- $-1$: client'ın o sample için abstain etmesi, yani geçerli tahmin vermemesi

Dawid–Skene, her round bu matris üzerinde MAP-EM çalıştıracak ve şu değerleri tahmin edecektir:

- Her sample'ın gerçek sınıfına ilişkin posterior olasılıkları
- Sınıf öncül olasılıkları
- Her client'ın her sınıf için confusion matrix'i

Sample $i$ için yayınlanacak label:

$$
\hat{y}^{DS}_i = \arg\max_c P(z_i=c \mid A)
$$

Server client'lara şunları broadcast edecektir:

- DS posterior argmax hard label'ları
- Common valid mask

Bu kolda kesinlikle MV fallback olmayacaktır.

#### EM yakınsamazsa ne olacak?

Maksimum EM iterasyon sayısı:

```yaml
dawid_skene_max_iterations: 500
```

500 iterasyon sonunda tolerance koşulu sağlanmamış olsa bile aşağıdakiler geçerliyse son DS sonucu kullanılacaktır:

- Posterior değerleri finite ise
- Prior değerleri finite ise
- Confusion matrix değerleri finite ise
- Olasılık satırlarının toplamı 1 ise
- Objective hesaplanabiliyorsa

Round şu şekilde işaretlenecektir:

```text
not_converged_but_used
```

Yani DS-only kolunda “yakınsamadı, o hâlde MV kullanalım” denmeyecektir. Son geçerli DS posterior'u broadcast edilecektir.

Eğer NaN, sonsuz değer veya bozuk normalizasyon gibi gerçek bir sayısal hata oluşursa MV kullanılmayacaktır. Run hata vererek durdurulacaktır. Hata düzeltildikten sonra karşılaştırmanın tamamı aynı kod sürümüyle yeniden çalıştırılacaktır.

#### Majority Vote DS-only içinde kullanılacak mı?

Training kararında kullanılmayacaktır.

MV:

- DS output'unu reddetmek için kullanılmayacaktır.
- Fallback için kullanılmayacaktır.
- Broadcast edilecek label'ı değiştirmeyecektir.
- Valid mask'i değiştirmeyecektir.

MV karşılaştırması yalnızca run sonrasında offline analizde hesaplanabilir.

Önerilen yeni config değeri:

```yaml
ssfl_hard_aggregation: dawid_skene_only
```

Bu mod mevcut kodda henüz bulunmamaktadır ve run öncesinde eklenmesi gerekmektedir.

### 4.3. Hybrid

Hybrid kolunda aynı round için hem MV hem DS adayı bulunacaktır.

DS adayı aşağıdaki koşulları sağlarsa DS label'ları broadcast edilecektir:

1. En az 3 uygun client bulunması
2. Annotation bulunması
3. EM'in en fazla 500 iterasyonda yakınsaması
4. Posterior, prior ve confusion değerlerinin finite olması
5. Normalizasyon kontrollerinin geçmesi
6. Objective'in bozulmaması
7. Sınıf kimliği ve tutarlılık kontrollerinin geçmesi
8. Majority Vote ile agreement değerinin en az 0.5 olması
9. Diagonal-ratio değerinin belirlenen 0.7 eşiğini geçmesi

Bu kontrollerden biri başarısızsa o round'da MV sonucu broadcast edilecektir.

Önemli ayrım:

> Hybrid algoritma güvenilir olmayan tek tek client'ları reddetmeyecektir. DS, uygun client'ların tamamını confusion matrix'leri üzerinden farklı ağırlıklarla değerlendirecektir. Fallback durumunda reddedilen şey tek bir client değil, o round'ın DS sonucunun tamamı olacaktır.

Önerilen config değeri:

```yaml
ssfl_hard_aggregation: dawid_skene
```

Hybrid ve DS-only'de aynı DS ayarları kullanılmalıdır. Özellikle her ikisinde de maksimum EM iterasyonu 500 olmalıdır. Böylece iki kol arasındaki temel fark DS estimator'ü değil, fallback politikası olur.

## 5. DS sınıf kimliği problemi

Dawid–Skene'in latent class index'lerinin gerçek class index'leriyle eşleşmesi garanti edilmelidir. Örneğin latent class 0'ın gerçekten `benign` anlamına geldiğinden emin olunmalıdır.

DS-only kolunda bunu MV'ye bağlı bir kabul/red kontrolüyle yapmak doğru olmaz. Bu nedenle önerilen yöntem:

1. EM başlangıcı mevcut class index'lerine göre yapılacaktır.
2. EM sonunda latent class'lar ile client'ların kullandığı label alanı arasında deterministik bir eşleştirme kurulacaktır.
3. Eşleştirme, toplam confusion uyumunu en yüksek yapan permutation ile bulunacaktır.
4. Eşit skor varsa deterministik olarak en küçük indeksli eşleştirme seçilecektir.
5. Ground-truth veya test label'ları kesinlikle kullanılmayacaktır.
6. MV, DS-only output'unu kabul etmek veya reddetmek için kullanılmayacaktır.

Bu alignment hem DS-only hem Hybrid için ortak olmalıdır. Böylece Hybrid'e verilen DS adayı ile DS-only'nin kullandığı DS adayı aynı olur.

## 6. Valid mask'in sabit tutulması

Karşılaştırmada önemli bir karıştırıcı değişken, yöntemlerin farklı sayıda open sample kullanması olabilir.

Bunu engellemek için üç kolda da sample şu koşulla valid sayılacaktır:

$$
\mathrm{valid}_i = \mathbf{1}(\text{sample } i \text{ için en az bir geçerli annotation var})
$$

DS için:

```yaml
dawid_skene_min_item_annotations: 1
dawid_skene_posterior_threshold: 0.0
```

Her round şu kontrol otomatik yapılmalıdır:

```text
MV valid mask == DS valid mask
```

Bit düzeyinde eşitlik sağlanmıyorsa karşılaştırma durdurulmalıdır. Çünkü bu durumda yöntemler aynı pseudo-label'ları değil, farklı sample kümelerini kullanıyor olur.

## 7. Ortak DS parametreleri

DS-only ve Hybrid için:

```yaml
dawid_skene_warmup_rounds: 0
dawid_skene_max_iterations: 500
dawid_skene_min_iterations: 2
dawid_skene_tolerance: 1.0e-6
dawid_skene_initialization_pseudocount: 0.01
dawid_skene_confusion_pseudocount: 0.1
dawid_skene_class_prior_pseudocount: 1.0
dawid_skene_min_item_annotations: 1
dawid_skene_min_client_annotations: 1
dawid_skene_min_clients: 3
dawid_skene_posterior_threshold: 0.0
dawid_skene_damping: 1.0
dawid_skene_epsilon: 1.0e-12
dawid_skene_warm_start: false
```

`min_client_annotations: 1` olması şu anlama gelir:

- En az bir geçerli annotation veren client DS hesabına katılır.
- Client, güvenilirlik skoru düşük olduğu için dışarı atılmaz.
- Tamamen abstain eden, yani kullanılabilir hiçbir bilgi göndermeyen client hesaplamaya katılamaz.
- Katılan client'lar sınıf bazındaki confusion matrix'lerine göre ağırlıklandırılır.

## 8. Her round kaydedilecek sonuçlar

### Model performansı

Her üç kolda, her round:

- Test accuracy
- Macro-F1
- Micro-F1
- Weighted-F1
- Her sınıf için precision
- Her sınıf için recall
- Her sınıf için F1
- Test confusion matrix

### Aggregation performansı

Ground-truth open label'lar training sırasında server tarafından kullanılmayacaktır. Yalnızca offline değerlendirmede:

- Broadcast pseudo-label accuracy
- Class-wise pseudo-label precision/recall/F1
- Valid rate
- All-abstain sayısı
- Tie sayısı
- Tie sample accuracy
- MV–DS disagreement sayısı
- MV–DS disagreement sample'larındaki accuracy
- Her sınıfa atanan pseudo-label sayısı

### DS tanı değerleri

DS-only ve Hybrid için her round:

- EM iteration sayısı
- `converged` bilgisi
- Stop reason
- Log-likelihood
- MAP objective
- Posterior confidence ortalaması
- Uygun client sayısı
- Dışarıda kalan all-abstain client sayısı
- Class alignment permutation'ı
- Class alignment skoru
- MV agreement — sadece raporlama amacıyla
- Diagonal fraction
- Her client'ın $11\times11$ confusion matrix'i

Hybrid için ayrıca:

- O round'da DS mi MV mi kullanıldığı
- Fallback yapıldıysa kesin nedeni
- Toplam DS kullanılan round sayısı
- Toplam MV fallback sayısı

DS-only için:

- `not_converged_but_used` round sayısı
- Gerçek sayısal hata sayısı
- Hiçbir round'da MV kullanılmadığını doğrulayan sayaç

## 9. Client güvenilirlik grafikleri

Client $j$'nin class $c$ için güvenilirliği:

$$
R_{j,c}^{(t)} = P(\text{client }j\text{ class }c\text{ der}\mid\text{latent gerçek class }c)
$$

Bu, DS confusion matrix'inin diagonal elemanıdır:

$$
R_{j,c}^{(t)} = \Theta_{j,c,c}^{(t)}
$$

Her round için bu değer kaydedilecektir.

Üretilecek grafikler:

1. **Private data class-count heatmap**
   - Satırlar: 89 client
   - Sütunlar: 11 class
   - Renk: client'ın private datasındaki örnek sayısı

2. **Class bazında güvenilirlik heatmap'leri**
   - Her class için ayrı panel
   - Satırlar: 89 client
   - Sütunlar: round 1–200
   - Renk: $R_{j,c}^{(t)}$

3. **Client başına class reliability grafikleri**
   - Her client için 11 class'ın zaman içindeki reliability değişimi
   - Çok kalabalık olacağı için ana raporda heatmap, ek materyalde client bazlı çizimler kullanılacaktır.

4. **Private sample sayısı–reliability ilişkisi**
   - X ekseni: client'ın ilgili class'taki private sample sayısı
   - Y ekseni: son 10 round ortalama reliability
   - Sınıfı hiç bulunmayan client'lar ayrı renkle gösterilecektir.

Bir client'ın private datasında belirli bir class yoksa bu değer grafikte sıfır güvenilirlik gibi yorumlanmamalıdır. Böyle hücreler:

- `unsupported` olarak işaretlenmeli
- Gri/boş gösterilmeli
- Ortalama hesaplarına yanlışlıkla sıfır olarak katılmamalıdır

## 10. Raw annotation kaydı

89 client'ın 8.900 sample için verdiği tahminler her round kaydedilmelidir. Bu kayıt sayesinde:

- Her yöntemin pseudo-label davranışı yeniden hesaplanabilir.
- DS sonuçları offline doğrulanabilir.
- Reliability grafikleri yeniden üretilebilir.
- MV run'ındaki client tahminleri üzerinde counterfactual DS analizi yapılabilir.

Bu matrisler client kimliğiyle ilişkili ayrıntılı bilgi içerdiği için restricted diagnostic olarak tutulmalıdır:

- Client kimlikleri pseudonymize edilmelidir.
- Dosyalar makale artifact'ına doğrudan konulmamalıdır.
- Grafik ve özet tablolar üretildikten sonra ham annotation'lar güvenli şekilde kaldırılmalı veya erişimi sınırlandırılmalıdır.

## 11. Run öncesi zorunlu testler

Tam 200-round run'lar başlamadan önce aşağıdaki kontroller geçmelidir.

### Kod testleri

- MV-only'nin mevcut Majority Vote çıktısını bit düzeyinde koruduğu
- DS-only'nin hiçbir kod yolunda MV label broadcast etmediği
- DS-only'de `not_converged` durumunda finite son posterior'un kullanıldığı
- Hybrid'de aynı durumda MV fallback yapıldığı
- DS-only'de sayısal hatanın fallback yerine run failure oluşturduğu
- Hybrid'de sayısal hatanın doğru fallback nedeni ile kaydedildiği
- Aynı DS candidate'ın DS-only ve Hybrid'e verildiği
- Class permutation/alignment testinin doğru çalıştığı
- Üç yöntem arasındaki valid mask'in eşit olduğu
- Dawid–Skene'in global random state tüketmediği
- Client upload sırasının sonucu değiştirmediği

### İki round smoke test

Her yöntem için Scenario 3 üzerinde iki round çalıştırılacaktır:

1. MV smoke
2. DS-only smoke
3. Hybrid smoke

Bu smoke testlerin amacı performansa bakmak değildir. Yalnızca aşağıdakiler doğrulanacaktır:

- 89 client'ın tamamının katıldığı
- GPU kullanımının doğru olduğu
- Dosyaların yazıldığı
- Metric kolonlarının eksiksiz olduğu
- DS-only'de fallback olmadığı
- Hybrid fallback nedenlerinin kaydedildiği
- Reliability matrislerinin üretildiği
- Bir round'ın yaklaşık çalışma süresi

Smoke sonuçlarına bakılarak DS parametresi değiştirilmeyecektir.

## 12. Tam run sırası

Smoke testleri geçtikten sonra kod ve config dondurulacaktır.

Tam run'lar tek GPU üzerinde sırayla çalıştırılacaktır:

1. `experiment1_s3_mv_200`
2. `experiment1_s3_ds_only_200`
3. `experiment1_s3_hybrid_200`

Run'lar paralel çalıştırılmayacaktır. Mevcut ayarlarda GPU zaten sekiz eşzamanlı client tarafından kullanılmaktadır.

Her run başlangıcında kaydedilecek bilgiler:

- Git commit hash
- Dataset manifest hash
- Resolved config
- Python ve dependency sürümleri
- CUDA sürümü
- GPU modeli
- Başlangıç zamanı
- Seed
- Scenario
- Client sayısı

Her round'da beklenen:

```text
num_proposals = 89
rejected_count = 0
```

Bir client mesajı eksik veya geçersizse run geçersiz sayılmalı ve sorun araştırılmalıdır.

## 13. Ana değerlendirme ölçütü

### Primary metric

Son 10 round'ın ortalama test accuracy değeri:

$$
\overline{\mathrm{Accuracy}}_{191:200}
=
\frac{1}{10}
\sum_{t=191}^{200}
\mathrm{Accuracy}_t
$$

Yalnızca round 200'ü kullanmak yerine son 10 round ortalaması alınması, tek bir round'daki dalgalanmanın sonucu belirlemesini engeller.

### Önemli secondary metric

Son 10 round'ın ortalama Macro-F1 değeri:

$$
\overline{\mathrm{MacroF1}}_{191:200}
$$

Macro-F1 özellikle önemlidir çünkü Scenario 3'te sınıf dağılımı heterojendir ve bazı device'larda Mirai sınıfları bulunmamaktadır.

### Destekleyici metrikler

- Round 200 accuracy
- Round 200 Macro-F1
- 200 round boyunca accuracy eğrisi
- 200 round boyunca Macro-F1 eğrisi
- Öğrenme eğrisi alanı
- Her class için son 10 round recall/F1
- Pseudo-label accuracy
- MV–DS disagreement sample performansı
- Tie sample performansı

`Best round` ana sonuç olarak kullanılmayacaktır.

## 14. Sonuçların yorumlanması

### DS-only, MV-only'den daha iyiyse

DS'nin client güvenilirliklerini modellemesinin Scenario 3'te fayda sağladığı söylenebilir.

Bu yorumun güçlü olması için:

- Accuracy artmalıdır.
- Macro-F1 artmalıdır.
- Artış yalnızca tek bir sınıftan kaynaklanmamalıdır.
- Mirai sınıflarında ciddi bir çöküş olmamalıdır.
- Valid sample sayısı aynı kalmalıdır.

### DS-only accuracy artırıp Macro-F1 düşürürse

Sonuç “DS daha iyi” şeklinde verilmemelidir. DS bazı büyük sınıfları iyileştirirken az temsil edilen sınıflara zarar veriyor olabilir.

### Hybrid iki saf yöntemden de iyiyse

Hybrid fallback algoritmasının katkısı savunulabilir:

$$
\mathrm{Hybrid} > \max(\mathrm{MV-only},\mathrm{DS-only})
$$

Bu karşılaştırma hem accuracy hem Macro-F1 için yapılmalıdır.

### Hybrid yalnızca DS-only'den iyi fakat MV-only'den kötüyse

Fallback mekanizması DS'nin zararını azaltıyor demektir; ancak MV'nin üzerine bir improvement sağladığı iddia edilemez.

### DS-only ile Hybrid neredeyse aynıysa

Hybrid fallback mekanizmasının ek katkısı sınırlı olabilir veya fallback çok az tetiklenmiş olabilir.

### MV-only en iyi sonucu verirse

Scenario 3'te DS'nin güvenilirlik modellemesinin yeterli fayda sağlamadığı ya da DS'nin sınıf tanımlanabilirliği problemi yaşadığı sonucuna gidilir. Sonuç saklanmamalı veya sadece başarısız round'lar çıkarılarak raporlanmamalıdır.

## 15. Hazırlanacak nihai grafikler

Ana raporda şu grafikler bulunmalıdır:

1. Üç yöntemin 200-round test accuracy eğrisi
2. Üç yöntemin 200-round Macro-F1 eğrisi
3. Son 10 round accuracy ve Macro-F1 karşılaştırma grafiği
4. 11 sınıf için per-class recall karşılaştırması
5. 11 sınıf için per-class F1 karşılaştırması
6. Broadcast pseudo-label accuracy eğrisi
7. MV–DS disagreement oranı
8. Tie sayısı ve tie sample accuracy eğrisi
9. DS EM iteration sayısı
10. DS convergence/nonconvergence timeline
11. Hybrid DS/MV seçim timeline'ı
12. Hybrid fallback nedenleri
13. Client × class private sample-count heatmap'i
14. Her class için client × round reliability heatmap'i
15. Private class sayısı ile reliability arasındaki ilişki
16. Device 3 ve Device 7'nin Mirai sınıflarındaki davranışını gösteren ayrı analiz

Accuracy grafiğinde ham değerler ve yalnızca görsel okunabilirlik için 10-round hareketli ortalama birlikte gösterilebilir. İstatistiksel hesaplar ham metriklerden yapılacaktır.

## 16. İstatistiksel sınırlama

Bu ilk aşamada her yöntem yalnızca seed 2023 ile bir kez çalıştırılacağı için üç run, üç bağımsız istatistiksel tekrar değildir.

Ayrıca 200 round, 200 bağımsız gözlem olarak kabul edilemez; birbirini takip eden round'lar güçlü biçimde bağlantılıdır. Bu nedenle:

- Round'ları bağımsız sample kabul ederek p-value hesaplanmamalıdır.
- İlk sonuçlar eşleştirilmiş, tek-seed deney olarak raporlanmalıdır.
- Makale düzeyinde güçlü sonuç gerekiyorsa aynı üçlü deney daha sonra ek seed'lerle tekrarlanmalıdır.
- Ancak bu ek tekrarlar mevcut üç run tamamlandıktan ve teknik doğruluk teyit edildikten sonra planlanmalıdır.

## 17. Run başlamadan önce tamamlanması gerekenler

Mevcut kod doğrudan bu deneyi çalıştırmaya hazır değildir. Şu anki `dawid_skene` modu aslında Hybrid davranışıdır. DS başarısızsa MV broadcast etmektedir.

Tam run öncesinde:

1. Yeni `dawid_skene_only` modu eklenmelidir.
2. Nonconverged ama sayısal olarak geçerli son DS posterior'u korunmalıdır.
3. DS estimator ile Hybrid fallback politikası birbirinden ayrılmalıdır.
4. DS-only'nin MV kullanmadığı testlerle kanıtlanmalıdır.
5. DS class alignment yöntemi eklenmelidir.
6. Her round client confusion/reliability değerleri kaydedilmelidir.
7. Üç kol için Scenario 3 config'leri oluşturulmalıdır.
8. Üç adet iki-round smoke test geçmelidir.
9. Kod ve config dondurulmalıdır.
10. Bundan sonra 600 round'luk tam deney başlatılmalıdır.

## 18. Run öncesinde onaylanacak iki kritik karar

- DS-only ve Hybrid için aynı `max_iterations: 500` kullanılacaktır.
- DS-only'de MV'ye bağlı kabul/red kontrolü yerine deterministik, ground-truth kullanmayan class alignment uygulanacaktır.

Bu iki karar onaylanmadan implementasyon veya run başlatılmamalıdır.

