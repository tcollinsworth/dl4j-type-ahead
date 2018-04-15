package com.daisyworks.demo.language;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author troy
 *
 */
public class DataSwizzler {
	private static final String inputFilename = "src/main/resources/examples/bookTitlesRaw500.txt";

	private static final String lineOutputFilename = "src/main/resources/examples/bookTitles.txt";
	private static final String ngramOutputFilename = "src/main/resources/examples/ngrams.txt";
	private static final String wordStatsOutputFilename = "src/main/resources/examples/wordStats.txt";
	private static final String firstCharWordMapOutputFilename = "src/main/resources/examples/firstCharWordMap.txt";
	private static final String charMapOutputFilename = "src/main/resources/examples/charMap.txt";
	private static final String maxLineLengthOutputFilename = "src/main/resources/examples/maxLineLength.txt";

	private final int min1gramWordLength;
	private final int minNgramWords;
	private final int maxNgramWords;

	private int examplesMaxCharLength;

	private final Map<String, Integer> wordStatsMap = new HashMap<>();
	private final Map<Character, List<String>> firstCharWordMap = new HashMap<>();

	private final Set<Integer> allCharInts = new HashSet<Integer>();

	// key = char (int), val = char sorted order index from 0-n for all characters
	// in training set
	// encoded to uniform space eliminating gaps and outliers
	// private static final Map<Character, Double> charMap = new HashMap<Character, Double>();

	public DataSwizzler(int min1gramWordLength, int minNgramWords, int maxNgramWords) {
		this.min1gramWordLength = min1gramWordLength;
		this.minNgramWords = minNgramWords;
		this.maxNgramWords = maxNgramWords;
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		DataSwizzler ds = new DataSwizzler(3, 1, 3);

		try (BufferedReader r = new BufferedReader(new FileReader(new File(inputFilename))); //
				BufferedWriter lw = new BufferedWriter(new FileWriter(new File(lineOutputFilename))); //
				BufferedWriter ngw = new BufferedWriter(new FileWriter(new File(ngramOutputFilename))); //
				BufferedWriter wmw = new BufferedWriter(new FileWriter(new File(wordStatsOutputFilename))); //
				BufferedWriter fcwmw = new BufferedWriter(new FileWriter(new File(firstCharWordMapOutputFilename))); //
				BufferedWriter cmw = new BufferedWriter(new FileWriter(new File(charMapOutputFilename))); //
				BufferedWriter llw = new BufferedWriter(new FileWriter(new File(maxLineLengthOutputFilename))); //
		) {
			String line;
			while ((line = r.readLine()) != null) {
				if (line.trim().isEmpty()) {
					continue;
				}
				line = ds.lineTransforms(line);
				line = ds.charTransforms(line);
				if (ds.examplesMaxCharLength < line.length()) {
					ds.examplesMaxCharLength = line.length();
				}
				ds.writeLineOutput(line, lw);
				ds.accumulateUniqueChars(line);
				ds.wordProcessing(line, ngw);
			}
			ds.saveLineLength(llw);
			ds.saveWordMaps(wmw, fcwmw);
			ds.saveCharMap(cmw);
		}
	}

	private void saveLineLength(BufferedWriter llw) throws IOException {
		llw.write(String.format("%d", examplesMaxCharLength));
	}

	private void accumulateUniqueChars(String sequence) {
		sequence.chars().forEach((c) -> allCharInts.add(c));
	}

	private void saveCharMap(BufferedWriter cmw) throws IOException {
		Integer[] uniquChars = allCharInts.toArray(new Integer[0]);
		Arrays.sort(uniquChars);

		// cmw.write(Arrays.toString(uniquChars));
		// cmw.write('\n');
		// Create and persist map of chars for input vector - needs to be reusable.
		// 10 - 255, then all higher map as discovered

		int[] i = { 0 };
		// Arrays.asList(uniquChars).stream().filter((c) -> c > 255).forEach((c) ->
		// persistChar(++i[0], c, w));
		Arrays.asList(uniquChars).forEach((c) -> persistChar(i[0]++, c, cmw));
		// cmw.write('\n');
	}

	private void persistChar(int i, Integer c, BufferedWriter w) {
		try {
			// w.write(String.format("%d:%d\n", i - 10, c));
			w.write(String.format("%d:%d:%c\n", i, c, c));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private void saveWordMaps(BufferedWriter wmw, BufferedWriter fcwmw) {
		// List<Map.Entry<String, Integer>> words =
		wordStatsMap.entrySet() //
				.stream() //
				.sorted((e1, e2) -> Integer.compare(e2.getValue(), e1.getValue())) //
				.collect(Collectors.toList()) //
				.forEach(e -> {
					accumulateFistCharWordList(e.getKey());
					try {
						wmw.write(String.format("%d\t%s\n", e.getValue(), e.getKey()));
					} catch (Exception e3) {
						e3.printStackTrace();
					}
				});
		firstCharWordMap.entrySet() //
				.stream() //
				.sorted((e1, e2) -> Character.compare(e1.getKey(), e2.getKey())) //
				.forEach(e -> {
					List<String> sortedWords = e.getValue() //
							.stream() //
							.sorted((w1, w2) -> w1.compareTo(w2)) //
							.collect(Collectors.toList());
					try {
						fcwmw.write(String.format("%c:%s\n", e.getKey(), Arrays.toString(sortedWords.toArray(new String[0]))));
					} catch (Exception e3) {
						e3.printStackTrace();
					}
				});
	}

	private void accumulateFistCharWordList(String word) {
		if (word.isEmpty() || word.length() < 1) {
			return;
		}
		List<String> words = firstCharWordMap.get(word.charAt(0));
		if (words == null) {
			words = new ArrayList<>();
			firstCharWordMap.put(word.charAt(0), words);
		}
		words.add(word);
	}

	private void wordProcessing(String line, BufferedWriter ngw) {
		String[] words = line.split(" ");

		List<String> wordsOrdered = Arrays.asList(words);

		wordsOrdered.stream().forEach(w -> {
			Integer wordCount = wordStatsMap.get(w);
			if (wordCount == null) {
				wordStatsMap.put(w, 1);
			} else {
				wordStatsMap.put(w, ++wordCount);
			}
		});

		Set<String> ngrams = getNgrams(wordsOrdered, minNgramWords, maxNgramWords);
		ngrams.stream().forEach(ngram -> {
			// DO NOT save/train on 1-gram words less than 3 characters
				String[] ngramWords = ngram.split(" ");
				if (ngramWords.length < 2 && ngramWords[0].length() < min1gramWordLength) {
					return; // continue
				}
				try {
					ngw.write(ngram);
					ngw.write('\n');
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
	}

	private String charTransforms(String line) {
		line = line.toLowerCase();
		line = line.replaceAll("[:();\\[\\]!?{}<>&@,®]", "");
		line = line.replaceAll("  ", " "); // remove all double spaces
		line = line.replaceAll("/", " ");
		line = line.replaceAll("\\. ", " ");
		line = line.replaceAll("\\.$", "");
		line = line.replaceAll("—", "-");
		line = line.replaceAll("’", "'");
		line = line.replaceAll("\u00A0", " "); // non-breaking space
		return line;
	}

	private void writeLineOutput(String line, BufferedWriter lw) throws IOException {
		if (!line.isEmpty()) {
			lw.write(line);
			lw.write('\n');
		}
	}

	/**
	 * Strip prefix and opening quote <br>
	 * Strip trailing quote
	 * 
	 * @return update line
	 */
	private String lineTransforms(String line) {
		final String prefix = "\"bookTitle\": \"";
		int beginIndex = line.indexOf(prefix) + prefix.length();
		int endIndex = line.lastIndexOf('"');
		return line.substring(beginIndex, endIndex).trim();
	}

	private final Set<String> getNgrams(List<String> wordListOrdered, int minNgramWords, int maxNgramWords) {
		Set<String> ngrams = new HashSet<String>();
		SentenceNgramTokenizer.tokenize(wordListOrdered, minNgramWords, maxNgramWords, ngrams);
		// WordNGramTokenizer.tokenize(wordListOrdered, minNgramWords, maxNgramWords,
		// ngrams);
		return ngrams;
	}
}
