FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /workspace

# Copy pom trước để cache dependency khi chỉ đổi source
COPY pom.xml .
RUN --mount=type=cache,target=/root/.m2 mvn -B -q dependency:go-offline

COPY src ./src
RUN --mount=type=cache,target=/root/.m2 mvn -B -q -DskipTests package


FROM eclipse-temurin:21-jre
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --no-create-home app

# Model ONNX (~240MB): PaddleOCR PP-OCRv5 detection + VietOCR vgg_transformer. Đặt trước COPY jar để cache layer
RUN mkdir -p /app/models \
    && curl -fsSL -o /app/models/det.onnx https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx \
    && for f in vgg_encoder.onnx vgg_encoder.onnx.data vgg_decoder.onnx vocab.json; do \
         curl -fsSL -o /app/models/$f https://huggingface.co/vemines/vietocr-onnx/resolve/main/$f; \
       done

COPY --from=build /workspace/target/*.jar app.jar
# Model vie+eng đầy đủ (chính xác hơn bản apt tesseract-ocr-vie)
COPY tessdata ./tessdata

USER app
ENV OCR_TESSDATA_PATH=/app/tessdata \
    OCR_MODELS_PATH=/app/models \
    JAVA_OPTS="-XX:MaxRAMPercentage=75"
EXPOSE 8080

ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]
