package com.example.fileservice.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.QueueBuilder;
import org.springframework.amqp.rabbit.config.RetryInterceptorBuilder;
import org.springframework.amqp.rabbit.config.SimpleRabbitListenerContainerFactory;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.support.converter.JacksonJsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.boot.amqp.autoconfigure.SimpleRabbitListenerContainerFactoryConfigurer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

@Configuration
public class RabbitMQConfig {

    public static final String OCR_EXCHANGE = "file-service.ocr";
    public static final String OCR_ROUTING_KEY = "ocr.jobs";
    public static final String OCR_QUEUE = "file-service.ocr.jobs";

    public static final String OCR_DEAD_EXCHANGE = "file-service.ocr.dead";
    public static final String OCR_DEAD_QUEUE = "file-service.ocr.jobs.dead";

    // Exchange gửi job embed sang ai-service (ai-service tự khai báo queue của nó)
    public static final String EMBED_EXCHANGE = "file-service.embed";
    public static final String EMBED_ROUTING_KEY = "embed.jobs";

    // Chặn vòng lặp vô hạn khi một file làm ai-service chết trước khi ack (message cứ được giao lại); hết lượt thì vào dead-letter queue
    private static final int MAX_DELIVERIES = 5;

    @Bean
    public DirectExchange ocrExchange() {
        return new DirectExchange(OCR_EXCHANGE, true, false);
    }

    @Bean
    public DirectExchange ocrDeadExchange() {
        return new DirectExchange(OCR_DEAD_EXCHANGE, true, false);
    }

    @Bean
    public Queue ocrQueue() {
        return QueueBuilder.durable(OCR_QUEUE)
                .quorum()
                .deliveryLimit(MAX_DELIVERIES)
                .deadLetterExchange(OCR_DEAD_EXCHANGE)
                .deadLetterRoutingKey(OCR_ROUTING_KEY)
                .build();
    }

    @Bean
    public Queue ocrDeadQueue() {
        return QueueBuilder.durable(OCR_DEAD_QUEUE).build();
    }

    @Bean
    public Binding ocrBinding() {
        return BindingBuilder.bind(ocrQueue()).to(ocrExchange()).with(OCR_ROUTING_KEY);
    }

    @Bean
    public Binding ocrDeadBinding() {
        return BindingBuilder.bind(ocrDeadQueue()).to(ocrDeadExchange()).with(OCR_ROUTING_KEY);
    }

    @Bean
    public DirectExchange embedExchange() {
        return new DirectExchange(EMBED_EXCHANGE, true, false);
    }

    @Bean
    public MessageConverter messageConverter() {
        return new JacksonJsonMessageConverter();
    }

    // Retry vô hạn: hết lượt retry khi DB đang sập thì message bị bỏ và tài liệu kẹt PROCESSING
    @Bean
    public SimpleRabbitListenerContainerFactory deadLetterListenerContainerFactory(
            SimpleRabbitListenerContainerFactoryConfigurer configurer, ConnectionFactory connectionFactory) {
        SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
        configurer.configure(factory, connectionFactory);
        factory.setAdviceChain(RetryInterceptorBuilder.stateless()
                .configureRetryPolicy(policy -> policy
                        .maxRetries(Long.MAX_VALUE)
                        .delay(Duration.ofSeconds(2))
                        .multiplier(2)
                        .maxDelay(Duration.ofSeconds(30)))
                .build());
        return factory;
    }
}
