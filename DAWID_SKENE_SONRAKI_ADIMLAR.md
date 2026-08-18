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
