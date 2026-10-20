## Build stage
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn clean package -DskipTests

## RUN stage
FROM eclipse-temurin:21-jre
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
# Mở cả 2 cổng: 8081 (REST) và 9091 (gRPC)
EXPOSE 8081 9091
ENTRYPOINT ["java", "-jar", "app.jar"]