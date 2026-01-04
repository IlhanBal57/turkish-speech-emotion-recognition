import os
import argparse
import random
import asyncio
import subprocess
import shutil
import tempfile

import edge_tts

BASE_DEFAULT = r"C:\Users\IlhanBal\Desktop\İlhan\sen4107-training\sen4107\data\turkish_emotions"

# edge-tts validasyonları: rate mutlaka +/-% formatında olmalı, pitch de genelde +/-(Hz)
EMOTION_CFG = {
    "mutlu":    {"voice": "tr-TR-EmelNeural",  "rate": "+28%", "pitch": "+16Hz"},
    "uzgun":    {"voice": "tr-TR-EmelNeural",  "rate": "-35%", "pitch": "-22Hz"},
    "kizgin":   {"voice": "tr-TR-AhmetNeural", "rate": "+38%", "pitch": "+10Hz"},
    "notr":     {"voice": "tr-TR-AhmetNeural", "rate": "+0%",  "pitch": "+0Hz"},
    "igrenme":  {"voice": "tr-TR-EmelNeural",  "rate": "-10%", "pitch": "-8Hz"},
    "korku":    {"voice": "tr-TR-EmelNeural",  "rate": "+14%", "pitch": "+30Hz"},
    "saskin":   {"voice": "tr-TR-AhmetNeural", "rate": "+22%", "pitch": "+20Hz"},
}

# Her duyguya ÖZEL, çok belirgin, uzun metin havuzu (12 adet/duygu)
TEXTS = {
    "mutlu": [
        "Harika! Bugün inanılmaz mutluyum! İçim kıpır kıpır ve yüzüm sürekli gülüyor. Güzel haberler aldım; her şey yolunda gidiyor. Şu an neşem çok belli!",
        "Çok güzel! Şu an gerçekten çok neşeliyim. Enerjim yüksek, moralim yerinde. Konuşurken bile daha canlıyım. İçimde sıcak bir sevinç var!",
        "Vay be! Bugün her şey tam istediğim gibi ilerliyor. İçim umut dolu ve kendimi çok iyi hissediyorum. Gülümsememi tutamıyorum!",
        "Yaşasın! Şu an çok mutlu hissediyorum. Küçük şeyler bile beni sevindiriyor; sanki gün ışığı içime doldu. Neşem dışarı taşıyor!",
        "İnanılmaz! Bugün içimde tatlı bir sevinç var. Her şey daha kolay geliyor; sanki ekstra bir gücüm var. Mutluluğum çok belli!",
        "Harika bir gün! Şu an keyfim çok yerinde. İyi hissettiğimde daha hızlı konuşuyorum ve daha çok gülümsüyorum. Neşeliyim!",
        "Çok mutluyum! Şu an sevinçten içim içime sığmıyor. Moralim yükseldi ve kendime güvenim arttı. Gerçekten harika hissediyorum!",
        "Bugün yüzüm gülüyor. İçimde temiz bir mutluluk var; sanki her şey olması gerektiği gibi. Şu an çok iyi hissediyorum!",
        "Şu an neşeliyim ve huzurluyum. Konuşurken bile sesim daha parlak çıkıyor. İyi şeyler olacak gibi hissediyorum; mutluyum!",
        "Bugün çok güzel geçti! İçimde sevinç var ve bunu saklayamıyorum. Her şey yolunda; ben de çok mutluyum!",
        "Şu an gerçekten keyifliyim. İyi hissetmek çok güzel; insanın içi açılıyor. Ben de içim açılmış gibi hissediyorum!",
        "Mutluyum! Şu an hayat daha renkli görünüyor. İçimde tatlı bir sevinç var. Gülümsüyorum ve çok iyi hissediyorum!",
    ],
    "uzgun": [
        "Bilmiyorum… Şu an kendimi çok üzgün hissediyorum. İçimde ağır bir boşluk var. Konuşmak bile zor geliyor; kelimeler boğazımda düğümleniyor.",
        "Gerçekten… bugün moralim çok bozuk. İçim daralıyor ve hiçbir şeye hevesim yok. Sesim bile düşmüş durumda; çok üzgünüm.",
        "Şu an… içimde bir hüzün var. Sanki her şey yavaşladı. Normalde kolay gelen şeyler bile zor geliyor. Kendimi çok yorgun hissediyorum.",
        "Açıkçası… bugün hiç iyi değilim. İçimde kırgınlık var ve bunu saklayamıyorum. Ne kadar uğraşsam da moralim düzelmiyor… üzgünüm.",
        "Şu an gözlerim dolacak gibi. İçimde bir ağırlık var ve geçmiyor. Sadece biraz sessizlik istiyorum; gerçekten üzgünüm.",
        "Bugün kalbim kırık gibi. İçim sıkışıyor ve nefesim daralıyor. Bir şeyler yolunda değil ve bu beni derinden üzüyor.",
        "Şu an kendimi yalnız hissediyorum. Konuşurken bile içim acıyor. Her şey çok ağır geliyor; sanki içimde bir düğüm var.",
        "İçimde tarifsiz bir hüzün var. Kendimi toparlamaya çalışıyorum ama olmuyor. Sesim bile bunu anlatıyor… üzgünüm.",
        "Şu an çok kırgınım. Gülümsemek gelmiyor içimden. Her şey üst üste gelmiş gibi ve bu beni gerçekten üzüyor.",
        "Bugün zor bir gün. İçimde umut azaldı gibi hissediyorum. Konuşmam yavaşladı; çünkü enerjim düştü. Üzgünüm.",
        "Şu an içim ağır. Kendimi iyi hissetmiyorum ve bunu saklayamıyorum. Sadece biraz dinlenmek istiyorum… üzgünüm.",
        "Üzgünüm. İçimde bir boşluk var ve bunu dolduramıyorum. Konuşurken bile sesim düşüyor; moralim çok bozuk.",
    ],
    "kizgin": [
        "Hayır! Bu şekilde olmaz! Kaç kere söyledim? Bu dikkatsizlik kabul edilemez! Şu an gerçekten sinirleniyorum; sabrım taştı!",
        "Yeter artık! Bu kadar umursamazlık olmaz! Açık konuşuyorum: bu yaptığın yanlış! Şu an çok kızgınım!",
        "Şaka mı bu? İnanılır gibi değil! Sürekli aynı hata, aynı bahane! Hayır! Kabul etmiyorum! Çok sinirliyim!",
        "Bak! Sınırımı zorluyorsun! Bu iş böyle yürümez! Şu an öfkeliyim ve bunu saklamıyorum!",
        "Bu kadarı fazla! Beni görmezden gelemezsin! Şu an gerçekten kızgınım! Bu iş burada bitmedi!",
        "Kesinlikle hayır! Bu davranış saygısızlık! Şu an sinirden kendimi zor tutuyorum! Yeter!",
        "Bu yaptığın beni çıldırtıyor! Sürekli tekrar ediyor! Hayır! Böyle devam edemez! Şu an çok kızgınım!",
        "Yeter! Duyuyor musun? Yeter! Bu sorumsuzluk kabul edilemez! Şu an öfkem çok açık!",
        "Hayır! Bu konu kapanmadı! Şu an sinirliyim ve bunun bir sonucu olacak! Bu kadar basit!",
        "Bak, tekrar söylüyorum: bu yanlış! Bu kadar dikkatsizlik olmaz! Şu an gerçekten çok kızgınım!",
        "Şaka gibi! Bu kadar olmaz! Beni bu noktaya getirdin! Şu an öfkeliyim! Yeter artık!",
        "Yeter! Bu bir hata değil, bu umursamazlık! Şu an çok kızgınım ve bunu açıkça söylüyorum!",
    ],
    "notr": [
        "Bilgi: Şu an durumu olduğu gibi anlatıyorum. Dosyalar üretilecek, isimlendirilecek ve ilgili klasöre kaydedilecek. Özel bir duygu eklemiyorum.",
        "Not: Şu an bilgilendirme yapıyorum. Süreç şu: metni seç, sesi üret, dosyayı kaydet ve arşiv oluştur. Nötr konuşuyorum.",
        "Açıklama: Şu an teknik bir özet veriyorum. Klasörler duygu isimleriyle ayrılacak, dosyalar standart adlandırılacak ve saklanacak.",
        "Bilgi: Şu an bir görev tanımı yapıyorum. Her duygu için belirli sayıda dosya üretilecek, dizine yazılacak ve istenirse paketlenecek.",
        "Not: Şu an genel bir açıklama yapıyorum. Komut çalıştığında medya dosyaları oluşturulacak ve belirtilen konuma kaydedilecek.",
        "Açıklama: Şu an süreç anlatıyorum. Çıktı dosyaları kaydedilecek. Klasör yapısı sabit kalacak. Bu bir bilgilendirmedir.",
        "Bilgi: Şu an iş akışını tarif ediyorum. Üretim tamamlanınca dosyalar kontrol edilecek ve arşive eklenecek.",
        "Not: Şu an sadece bilgi aktarıyorum. Duygusal vurgu yapmıyorum. Dosyalar düzenli şekilde saklanacak.",
        "Açıklama: Şu an sistem çıktısını anlatıyorum. Üretilen dosyalar dizine yazılır ve seçeneğe göre arşivlenir.",
        "Bilgi: Şu an adımları sıralıyorum. Önce üretim, sonra kayıt, ardından paketleme yapılır. Nötr şekilde konuşuyorum.",
        "Not: Şu an basit bir açıklama yapıyorum. Dosyalar aynı formatta olacak ve aynı kök klasörde tutulacak.",
        "Açıklama: Şu an tarafsız bir anlatım yapıyorum. Bu bir bilgi metnidir ve duygu içermez.",
    ],
    "igrenme": [
        "Iyy… bu hiç hoş değil. Şu an ciddi şekilde iğreniyorum. İçim kaldırmıyor; midem bulanıyor. Lütfen bunu benden uzak tut.",
        "Bu çok kötü… gerçekten tiksindim. Şu an yüzümü buruşturuyorum ve uzaklaşmak istiyorum. Çok rahatsız edici.",
        "Iyy, hayır… bunu görmek bile istemiyorum. Şu an iğrenme hissi çok yoğun; midem bulanıyor. Lütfen kapat şunu.",
        "Bu nasıl bir şey? Çok pis… Şu an tiksiniyorum ve dayanmakta zorlanıyorum. Hemen uzaklaşmak istiyorum.",
        "Iyy! Bu koku bile berbat. Şu an resmen iğreniyorum. Böyle bir şeye bakmak istemiyorum.",
        "Bu görüntü rahatsız edici… Şu an tiksindim. İçim kaldırmıyor; midem bulanıyor ve geriliyorum.",
        "Hayır, hayır… bu çok iğrenç. Şu an tüylerim diken diken oldu. Lütfen bunu benden uzaklaştır.",
        "Bu gerçekten mide bulandırıcı. Şu an iğrenme hissini saklayamıyorum. Yüzüm bile buna tepki veriyor.",
        "Iyy… bu hiç normal değil. Şu an tiksiniyorum ve uzaklaşmak istiyorum. Çok rahatsız oldum.",
        "Bu ne böyle? Çok kötü… Şu an iğreniyorum. Bu his çok güçlü; dayanmak istemiyorum.",
        "Iyy! Şu an içim kalktı. Bu kadar pis bir şey olamaz. Hemen uzak durmam lazım.",
        "Bu rahatsız edici… Şu an tiksindim, midem bulanıyor. Lütfen bunu bitirelim.",
    ],
    "korku": [
        "Bir dakika… Şu an korkuyorum. Kalbim hızlı atıyor. Sanki kötü bir şey olacakmış gibi hissediyorum. Etrafı dinliyorum ve tedirginim.",
        "Dur… içimde panik var. Güvende hissetmiyorum. Sesler bana çok yakın geliyor. Ne yapacağımı bilemiyorum; korkuyorum.",
        "Şey… şu an endişeliyim. Nefesim hızlandı ve elim ayağım titriyor. Sanki biri beni izliyor gibi; çok tedirginim.",
        "Off… içim ürperiyor. Bu durum beni korkutuyor. Bir an önce buradan uzaklaşmak istiyorum. Korkuyorum.",
        "Bir dakika… panikledim. Düşüncelerim karmakarışık. Kalbim küt küt atıyor. Şu an güvende olmak istiyorum.",
        "Şu an tedirginim. Bir şeylerin ters gittiğini hissediyorum. Etraf çok sessiz ama ben daha çok korkuyorum.",
        "Korkuyorum… Sesim titriyor. İçimde kötü bir his var. Tehlike yaklaşmış gibi; gözüm sürekli etrafta.",
        "Şu an korkudan düşünemiyorum. Nefesim sıklaştı. Bir şey olacak diye bekliyorum; çok tedirginim.",
        "Şu an panik var içimde. Her şey çok hızlı oldu. Kendimi güvende hissetmiyorum ve bu beni korkutuyor.",
        "Tedirginim… Kalbim çok hızlı atıyor. Ne olacağını bilmiyorum. Şu an korkum çok belli.",
        "Şu an ürperiyorum. Bir şey olacakmış gibi hissediyorum. Sesler ve gölgeler beni daha da korkutuyor.",
        "Korkuyorum… İçim sıkışıyor. Sadece güvenli bir yere gitmek istiyorum. Çok tedirginim.",
    ],
    "saskin": [
        "Ne? Bir saniye… Az önce ne oldu? Şu an çok şaşkınım. Beklemiyordum, aklım karıştı. Gerçekten inanamadım!",
        "Vay! Bu hiç beklediğim gibi değil. Şu an şaşırdım ve ne diyeceğimi bilemiyorum. Bir an durup düşünmem lazım.",
        "Nasıl yani? Ciddi misin? Şu an resmen şaşkınlıktan donakaldım. Böyle bir şey beklemiyordum!",
        "Bir dakika… Bu gerçek mi? Şu an çok şaşkınım. Kafam karıştı ve gözlerim açıldı; beklenmedik bir durum!",
        "Ne diyorsun? Şu an şaşırdım. Bir anda oldu ve ben hazırlıksız yakalandım. Gerçekten hayret ettim!",
        "Vay canına… Şu an şaşkınlık içindeyim. Beklediğim her şey değişti gibi. Ne diyeceğimi bulamıyorum.",
        "Ne oluyor? Şu an çok şaşkınım. Bu kadar hızlı olmasını beklemiyordum. Bir an anlamaya çalışıyorum.",
        "Ciddi misin? Şu an çok şaşırdım. Bir anda bütün plan değişti. Gerçekten hayret!",
        "Bir saniye… Böyle bir şey mümkün mü? Şu an şaşkınlıktan ağzım açık kaldı. Beklenmedik bir sürpriz!",
        "Vay! Şu an şaşırdım ve garip hissediyorum. Beklemiyordum. Bir an durup toparlanmam lazım.",
        "Ne? Şu an çok şaşkınım. İçimde bir ‘nasıl olur’ hissi var. Bu gerçekten sürpriz oldu!",
        "Bir dakika… Bu çok ilginç. Şu an şaşkınım ve kafam karışık. Beklenmedik bir şey oldu!",
    ],
}

PREFIXES = {
    "mutlu": ["Harika!", "Çok güzel!", "İnanılmaz!", "Yaşasın!"],
    "uzgun": ["Bilmiyorum…", "Şu an…", "Açıkçası…", "Gerçekten…"],
    "kizgin": ["Bak!", "Hayır!", "Yeter!", "Şaka mı bu?"],
    "notr":   ["Bilgi:", "Not:", "Açıklama:", "Süreç:"],
    "igrenme": ["Iyy!", "Off!", "Hayır…", "Cık!"],
    "korku": ["Bir dakika…", "Dur…", "Şey…", "Off…"],
    "saskin": ["Ne?!", "Vay!", "Bir saniye…", "Nasıl yani?"],
}

def pick_text(emotion: str) -> str:
    base = random.choice(TEXTS[emotion])
    if random.random() < 0.5:
        return f"{random.choice(PREFIXES[emotion])} {base}"
    return base

async def synth_mp3(text: str, out_mp3: str, cfg: dict):
    communicate = edge_tts.Communicate(
        text=text,
        voice=cfg["voice"],
        rate=cfg["rate"],
        pitch=cfg["pitch"],
    )
    await communicate.save(out_mp3)

def mp3_to_wav(mp3_path: str, wav_path: str, wav_rate: int = 16000):
    # 16kHz mono WAV (ML için yaygın). İstersen 44100 yapabiliriz.
    cmd = [
        "ffmpeg", "-y",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", str(wav_rate),
        "-c:a", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def make_archive(folder_path: str, out_no_ext: str) -> str:
    winrar = r"C:\Program Files\WinRAR\WinRAR.exe"
    if os.path.exists(winrar):
        rar_path = out_no_ext + ".rar"
        subprocess.run([winrar, "a", "-r", rar_path, folder_path], check=True)
        return rar_path
    else:
        return shutil.make_archive(out_no_ext, "zip", folder_path)

async def generate_emotion(emotion: str, count: int, base: str, speaker: str, do_archive: bool, keep_mp3: bool, wav_rate: int):
    cfg = EMOTION_CFG[emotion]
    out_dir = os.path.join(base, emotion)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, count + 1):
        text = pick_text(emotion)
        out_wav = os.path.join(out_dir, f"{emotion}_{i:02d}_{speaker}.wav")

        # geçici mp3 üret
        with tempfile.TemporaryDirectory() as tmp:
            tmp_mp3 = os.path.join(tmp, f"{emotion}_{i:02d}_{speaker}.mp3")
            print("Üretiliyor:", out_wav)
            await synth_mp3(text, tmp_mp3, cfg)
            mp3_to_wav(tmp_mp3, out_wav, wav_rate=wav_rate)

            if keep_mp3:
                out_mp3 = os.path.join(out_dir, f"{emotion}_{i:02d}_{speaker}.mp3")
                shutil.copyfile(tmp_mp3, out_mp3)

    print(f"✅ {emotion} tamam. Klasör: {out_dir}")

    if do_archive:
        arch = make_archive(out_dir, os.path.join(base, emotion))
        print(f"📦 {emotion} arşiv: {arch}")

async def main_async(args):
    all_emotions = ["mutlu", "uzgun", "kizgin", "notr", "igrenme", "korku", "saskin"]
    if args.emotion != "all":
        if args.emotion not in all_emotions:
            raise SystemExit("emotion yanlış. Geçerli: mutlu, uzgun, kizgin, notr, igrenme, korku, saskin, all")
        emotions = [args.emotion]
    else:
        emotions = all_emotions

    # ffmpeg var mı hızlı kontrol
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        raise SystemExit("FFmpeg bulunamadı. 'ffmpeg -version' çalışmıyor. FFmpeg kurup PATH'e eklemelisin.")

    for emo in emotions:
        await generate_emotion(
            emotion=emo,
            count=args.count,
            base=args.base,
            speaker=args.speaker,
            do_archive=args.archive,
            keep_mp3=args.keep_mp3,
            wav_rate=args.wav_rate,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", default="all", help="mutlu/uzgun/kizgin/notr/igrenme/korku/saskin/all")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--speaker", default="k1")
    parser.add_argument("--archive", action="store_true", help="WinRAR varsa RAR yoksa ZIP")
    parser.add_argument("--keep-mp3", action="store_true", help="WAV yanında MP3 de sakla")
    parser.add_argument("--wav-rate", type=int, default=16000, help="WAV sample rate (16000 önerilir)")
    args = parser.parse_args()

    random.seed(42)
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
