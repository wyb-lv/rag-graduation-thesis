package com.example.fileservice.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class InternalAccessConfig implements WebMvcConfigurer {

    private final String token;

    public InternalAccessConfig(@Value("${internal.token:}") String token) {
        this.token = token;
    }

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(new InternalAccessInterceptor(token)).addPathPatterns("/internal/**");
    }
}
