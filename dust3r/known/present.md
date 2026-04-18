# Kế hoạch trình bày DUSt3R (bản dễ hiểu, tiếng Việt có dấu)

Tài liệu này giúp bạn trả lời 3 câu hỏi:
1) Nên tập trung vào phần nào để báo cáo đúng trọng tâm.
2) Trình bày từng phần như thế nào để thầy dễ theo dõi.
3) Nội dung chi tiết có thể nói trong từng phần.

---

## 0) Mục tiêu của buổi trình bày

Thông điệp chính cần chốt:
- DUSt3R giải bài toán dựng lại cảnh 3D từ ảnh theo 2 bước lớn.
  - Bước 1: Dự đoán điểm 3D dày đặc từ từng cặp ảnh.
  - Bước 2: Ghép nhiều cặp ảnh thành một cảnh 3D thống nhất.
- Điểm hay của DUSt3R là kết hợp tốt giữa mô hình học sâu và tối ưu hình học.
- Kết quả thực tế: Có thể dựng đám mây điểm 3D từ bộ ảnh mà không cần quy trình SfM đầy đủ như trước.

---

## 1) Nên tập trung trình bày phần nào

Nên tập trung 6 phần sau, theo đúng mạch của hệ thống:

1. Bài toán và đầu vào/đầu ra
- Hệ thống nhận gì, trả ra gì.
- Vì sao bài toán khó.

2. Kiến trúc mô hình xử lý cặp ảnh
- Phần mã hóa ảnh.
- Phần giải mã kết hợp 2 ảnh.
- Phần đầu ra 3D.

3. Cách mô hình học và vai trò của độ tin cậy
- Hàm lỗi học 3D.
- Độ tin cậy được học như thế nào.

4. Từ cặp ảnh sang nhiều ảnh: ghép toàn cục
- Biểu diễn bằng đồ thị ảnh.
- Tối ưu để cảnh 3D thống nhất.

5. Kết quả, ưu điểm, hạn chế
- Điểm mạnh khi áp dụng thực tế.
- Các tình huống dễ lỗi.

6. Hướng mở rộng
- Các công trình liên quan và ý tưởng cải tiến.

---

## 2) Cách trình bày từng phần (khung 4 câu)

Với mỗi phần, bạn bám 4 câu sau:
1) Phần này giải quyết vấn đề gì?
2) Nó làm như thế nào?
3) Trong mã nguồn nằm ở đâu?
4) Ý nghĩa của phần này với toàn hệ thống là gì?

Chỉ cần giữ đúng khung này, bài trình bày sẽ rõ và chắc.

---

## 3) Gợi ý thời gian cho báo cáo 20-25 phút

- Slide 1 (1 phút): Động cơ và mục tiêu.
- Slide 2 (2 phút): Toàn cảnh quy trình DUSt3R.
- Slide 3-5 (6 phút): Mô hình xử lý cặp ảnh.
- Slide 6 (3 phút): Hàm lỗi và độ tin cậy.
- Slide 7-8 (5 phút): Ghép toàn cục nhiều ảnh.
- Slide 9 (2 phút): Kết quả minh họa.
- Slide 10 (2 phút): Ưu điểm và hạn chế.
- Slide 11 (2 phút): Kết luận và hướng phát triển.
- Dự phòng 2 phút để trả lời câu hỏi.

---

## 4) Nội dung chi tiết để trình bày

## Phần A - Bài toán và cách đặt vấn đề

### A1. Bản chất bài toán
Bạn có thể nói:
- Mục tiêu của DUSt3R là dựng lại hình học 3D từ ảnh RGB.
- Hệ thống nhận từng cặp ảnh để dự đoán điểm 3D, rồi ghép nhiều cặp lại thành một cảnh.

### A2. Đầu vào và đầu ra
Bạn có thể nói:
- Đầu vào cho mỗi lần chạy là 2 ảnh của cùng cảnh.
- Đầu ra là:
  - Bản đồ điểm 3D cho ảnh 1.
  - Bản đồ điểm 3D cho ảnh 2 (được đổi về cùng hệ quy chiếu với ảnh 1).
  - Bản đồ độ tin cậy cho các điểm.
- Khi chạy nhiều ảnh, hệ thống trả về đám mây điểm 3D của toàn cảnh.

### A3. Vì sao khó
Bạn có thể nói:
- Từ ảnh 2D suy ra 3D luôn có nhiều khả năng, không có một đáp án duy nhất ngay từ đầu.
- Có vùng bị che khuất, mờ, ít họa tiết nên dự đoán không đều chất lượng.
- Mỗi cặp ảnh tự nó chỉ đúng cục bộ, cần ghép toàn cục để đồng nhất.

---

## Phần B - Kiến trúc mô hình xử lý cặp ảnh (quan trọng nhất)

### B1. Phần mã hóa ảnh (encoder)
Thông điệp chính:
- DUSt3R dùng lại phần mã hóa của CroCo làm nền.

Bạn có thể nói:
- CroCo đã được huấn luyện trước với bài toán so sánh hai góc nhìn, nên học được đặc trưng có ích cho hình học.
- Trong DUSt3R, ảnh được đổi thành các mảnh nhỏ, rồi đi qua nhiều lớp mã hóa.

Ý nghĩa:
- Mô hình không chỉ học màu sắc hay kết cấu, mà học quan hệ giữa hai góc nhìn.

### B2. Tách ảnh thành mảnh và xử lý kích thước ảnh
Thông điệp chính:
- Trước khi vào mô hình, ảnh phải được chia thành các mảnh nhỏ.

Bạn có thể nói:
- Có hai cách xử lý ảnh theo tỉ lệ khung hình.
- Với ảnh dọc/ảnh ngang, hệ thống có cơ chế dùng kích thước thật để đặt đúng vị trí mảnh ảnh.

Ý nghĩa:
- Nếu đặt sai vị trí mảnh ảnh, mô hình sẽ hiểu sai quan hệ không gian.

### B3. Phần giải mã hai nhánh và tính bất đối xứng
Thông điệp chính:
- DUSt3R có hai nhánh giải mã chạy song song cho hai ảnh.

Bạn có thể nói:
- Sau khi mã hóa, hai bộ đặc trưng của ảnh 1 và ảnh 2 đi qua lớp đổi chiều đặc trưng.
- Tiếp theo là vòng lặp nhiều tầng giải mã.
- Mỗi tầng là một lần hai ảnh trao đổi thông tin hai chiều.

Câu dễ nhớ:
- Mỗi tầng giải mã là một lần hai ảnh nói chuyện với nhau để hiểu cảnh tốt hơn.

### B4. Phần đầu ra 3D
Thông điệp chính:
- DUSt3R không khôi phục lại ảnh, mà xuất trực tiếp điểm 3D.

Bạn có thể nói:
- Từ đặc trưng cuối, đầu ra chuyển thành bản đồ điểm 3D dày đặc và bản đồ độ tin cậy.
- Điểm 3D của ảnh 2 được đổi về hệ quy chiếu của ảnh 1 để tiện ghép.

Ý nghĩa:
- Kết quả cặp ảnh đã có thông tin hình học cần cho bước ghép toàn cục.

---

## Phần C - Cách học của mô hình và độ tin cậy

### C1. Hàm lỗi học 3D
Bạn có thể nói:
- Hàm lỗi chính đo sai khác giữa điểm 3D dự đoán và điểm 3D mục tiêu.
- Có biến thể để giảm phụ thuộc vào tỉ lệ tuyệt đối của cảnh.

### C2. Độ tin cậy
Bạn có thể nói:
- Độ tin cậy không phải đặt tay theo luật cứng.
- Mô hình tự học độ tin cậy cùng lúc khi học điểm 3D.
- Điểm nào dự đoán tốt thì độ tin cậy cao hơn.

Ý nghĩa:
- Độ tin cậy là trọng số quan trọng khi ghép nhiều ảnh và lọc nhiễu.

---

## Phần D - Ghép toàn cục từ nhiều cặp ảnh

### D1. Vì sao cần bước này
Bạn có thể nói:
- Mỗi cặp ảnh chỉ đúng trong phạm vi cặp đó.
- Muốn có một cảnh duy nhất, phải đưa tất cả về cùng một hệ tọa độ chung.

### D2. Biểu diễn bằng đồ thị
Bạn có thể nói:
- Mỗi ảnh là một nút.
- Mỗi cặp ảnh là một cạnh chứa ràng buộc hình học và độ tin cậy.

### D3. Khởi tạo và tối ưu
Bạn có thể nói:
- Hệ thống khởi tạo ban đầu bằng cây khung nhỏ nhất và ước lượng tư thế máy ảnh.
- Sau đó tối ưu lặp để giảm sai số toàn cục.

### D4. Làm sạch đám mây điểm
Bạn có thể nói:
- Dùng độ tin cậy và kiểm tra nhất quán nhiều góc nhìn để loại điểm kém.

Thông điệp chốt phần D:
- Đây là bước giúp DUSt3R làm tốt trên bộ ảnh nhiều góc nhìn, không chỉ từng cặp riêng lẻ.

---

## Phần E - Ưu điểm, hạn chế, bài học

### E1. Ưu điểm
- Quy trình rõ ràng: xử lý cặp ảnh trước, ghép toàn cục sau.
- Có bản đồ độ tin cậy để giảm ảnh hưởng điểm xấu.
- Dùng được cho nhiều bộ dữ liệu ảnh thực tế.

### E2. Hạn chế
- Nếu thiếu ràng buộc bổ sung, tỉ lệ tuyệt đối của cảnh có thể chưa chính xác.
- Dễ khó khi ảnh chồng lấp ít, ảnh mờ, hoặc vùng ít chi tiết.
- Bước ghép toàn cục phụ thuộc vào chất lượng khởi tạo và đồ thị cặp ảnh.

### E3. Bài học kỹ thuật
- Huấn luyện trước đúng bài toán giúp phần mã hóa học đúng loại thông tin cần thiết.
- Độ tin cậy là thành phần quan trọng để tối ưu ổn định.
- Tách rõ trách nhiệm từng tầng làm hệ thống dễ hiểu và dễ mở rộng.

---

## 5) Bản nói liền mạch (có thể đọc gần nguyên văn)

Em trình bày DUSt3R theo hai bước chính.

Bước thứ nhất là xử lý từng cặp ảnh: mô hình nhận hai ảnh, trích đặc trưng, cho hai ảnh trao đổi thông tin qua nhiều tầng, rồi xuất ra bản đồ điểm 3D dày đặc và bản đồ độ tin cậy.

Bước thứ hai là ghép toàn cục cho nhiều ảnh: mỗi cặp ảnh tạo một ràng buộc hình học trong đồ thị, sau đó bộ tối ưu tìm cách đặt tất cả ảnh về một hệ tọa độ chung để cảnh 3D nhất quán.

Điểm em thấy nổi bật là DUSt3R không dừng ở dự đoán cặp ảnh, mà có bước ghép toàn cục rõ ràng. Vì vậy kết quả cuối cùng dùng được tốt hơn cho dựng cảnh từ nhiều ảnh thực tế.

---

## 6) Câu hỏi thầy hay hỏi và cách trả lời ngắn

1. Tại sao cần CroCo?
- Vì phần mã hóa của CroCo đã học trước đặc trưng liên quan đến hai góc nhìn, nên khi chuyển sang DUSt3R sẽ học dựng 3D nhanh và ổn định hơn.

2. Tại sao cần độ tin cậy?
- Vì chất lượng dự đoán không đều theo từng điểm ảnh; độ tin cậy giúp giảm tác động của điểm kém khi tối ưu.

3. Tại sao xử lý cặp ảnh thôi chưa đủ?
- Vì mỗi cặp chỉ cho kết quả cục bộ; muốn có một cảnh thống nhất phải có bước ghép toàn cục.

4. Khác gì so với cách dựng 3D cổ điển?
- DUSt3R dự đoán điểm 3D dày đặc trực tiếp bằng học sâu, sau đó tối ưu toàn cục trên các ràng buộc đã học được.

---

## 7) Gợi ý hình nên có trong slide

- 1 hình luồng dữ liệu: ảnh -> mảnh ảnh -> mã hóa -> giải mã -> điểm 3D/độ tin cậy.
- 1 hình minh họa một tầng giải mã hai nhánh.
- 1 hình đồ thị ghép toàn cục (nút ảnh, cạnh ràng buộc).
- 1 hình trước/sau khi lọc và ghép toàn cục.

Nếu thiếu thời gian, ưu tiên 3 hình:
1) Kiến trúc xử lý cặp ảnh.
2) Hàm lỗi + độ tin cậy.
3) Ghép toàn cục.

---

## 8) Kết luận 3 câu để chốt bài

- DUSt3R kết hợp tốt giữa dự đoán 3D theo cặp ảnh và ghép toàn cục nhiều ảnh.
- Ba điểm kỹ thuật cốt lõi là: mã hóa từ CroCo, giải mã hai nhánh, và học độ tin cậy.
- Giá trị lớn nhất là tạo ra cảnh 3D thống nhất từ bộ ảnh thực tế, không chỉ xử lý từng cặp riêng lẻ.

---

## 9) Bảng giải thích từ khó (dùng khi thầy hỏi)

- Dự đoán dày đặc: Dự đoán gần như ở mọi điểm ảnh, không chỉ một số điểm đặc trưng.
- Bản đồ điểm 3D: Ảnh mà mỗi điểm ảnh đi kèm tọa độ 3D.
- Độ tin cậy: Mức độ mô hình tin rằng dự đoán ở điểm đó là đúng.
- Hệ quy chiếu: Cách chọn gốc tọa độ và trục tọa độ để biểu diễn điểm 3D.
- Bất đối xứng: Hai nhánh không hoàn toàn giống vai trò đầu ra, ảnh 2 được đổi về hệ của ảnh 1.
- Đồ thị ảnh: Cách mô tả nhiều ảnh bằng nút và cạnh để dễ tối ưu toàn cục.
- Cây khung nhỏ nhất: Cách chọn tập cạnh chính để khởi tạo ghép nhiều ảnh ổn định hơn.
- Tối ưu toàn cục: Tìm bộ tham số tốt nhất cho toàn bộ ảnh cùng lúc, không tối ưu từng cặp rời rạc.
