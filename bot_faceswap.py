import os
import cv2
import telebot
import insightface
from insightface.app import FaceAnalysis

TOKEN = "8871034169:AAFKpXeYQFU3Wni2Ka68UlVjXuWgGZA82rU"
bot = telebot.TeleBot(TOKEN)

FOLDER_HASIL = r"E:\SCRIPT AI\hasil"
if not os.path.exists(FOLDER_HASIL):
    os.makedirs(FOLDER_HASIL)

# Ubah ukuran deteksi agar lebih peka terhadap wajah kecil / kurang jelas
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)

swapper = insightface.model_zoo.get_model("inswapper_128.onnx", download=True)
user_photos = {}

@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "Halo! Kirimkan **Foto 1 (Target/Badan)** terlebih dahulu.")

@bot.message_handler(content_types=["photo"])
def handle_photos(message):
    chat_id = message.chat.id
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    if chat_id not in user_photos or len(user_photos[chat_id]) == 0:
        target_path = os.path.join(FOLDER_HASIL, f"target_{chat_id}.jpg")
        with open(target_path, "wb") as f:
            f.write(downloaded_file)
        user_photos[chat_id] = [target_path]
        bot.reply_to(message, "Foto Target diterima! Sekarang kirim **Foto 2 (Wajah Baru)**.")

    elif len(user_photos[chat_id]) == 1:
        source_path = os.path.join(FOLDER_HASIL, f"source_{chat_id}.jpg")
        with open(source_path, "wb") as f:
            f.write(downloaded_file)
        user_photos[chat_id].append(source_path)

        bot.reply_to(message, "Sedang menukar wajah... Harap tunggu sebentar.")

        try:
            target_img = cv2.imread(user_photos[chat_id][0])
            source_img = cv2.imread(user_photos[chat_id][1])

            target_faces = app.get(target_img)
            source_faces = app.get(source_img)

            if not target_faces:
                bot.send_message(chat_id, "Wajah pada Foto Pertama (Target) tidak terdeteksi. Gunakan foto target yang lebih jelas!")
            elif not source_faces:
                bot.send_message(chat_id, "Wajah pada Foto Kedua (Wajah Baru) tidak terdeteksi. Gunakan foto wajah yang lebih jelas!")
            else:
                res = target_img.copy()
                # Ambil wajah utama dari masing-masing foto
                res = swapper.get(res, target_faces[0], source_faces[0], paste_back=True)

                output_path = os.path.join(FOLDER_HASIL, f"result_{chat_id}.jpg")
                cv2.imwrite(output_path, res)

                with open(output_path, "rb") as foto_hasil:
                    bot.send_photo(chat_id, foto_hasil, caption="Berhasil! Ini hasil Face Swap kamu.")

        except Exception as e:
            bot.send_message(chat_id, f"Terjadi kesalahan: {str(e)}")

        user_photos[chat_id] = []

print("Bot Face Swap Lokal siap dijalankan...")
bot.infinity_polling()
