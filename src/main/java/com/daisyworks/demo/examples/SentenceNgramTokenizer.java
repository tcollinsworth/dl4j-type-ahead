package com.daisyworks.demo.examples;

import java.util.List;
import java.util.Set;

import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

public class SentenceNgramTokenizer {
	public static void tokenize(String sentence, int minWords, int maxWords, Set<String> ngrams) {
		TokenizerFactory factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), minWords, maxWords);
		List<String> tokens = factory.create(sentence).getTokens();
		ngrams.addAll(tokens);
	}

	public static void tokenize(List<String> wordListOrdered, int minWords, int maxWords, Set<String> ngrams) {
		StringBuilder sb = new StringBuilder();
		String[] delimiter = { "" };
		wordListOrdered.forEach((word) -> {
			sb.append(delimiter[0]).append(word);
			delimiter[0] = " ";
		});
		tokenize(sb.toString(), minWords, maxWords, ngrams);
	}
}
