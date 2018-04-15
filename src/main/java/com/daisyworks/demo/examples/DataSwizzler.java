package com.daisyworks.demo.examples;

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
	private static final String maxLineCharLengthOutputFilename = "src/main/resources/examples/maxLineCharLength.txt";
	private static final String maxNgramCharLengthOutputFilename = "src/main/resources/examples/maxNgramCharLength.txt";

	private final int min1gramWordLength;
	private final int minNgramWords;
	private final int maxNgramWords;

	private int examplesMaxCharLength;
	private int examplesMaxNgramCharLength;

	private final Map<String, Integer> wordStatsMap = new HashMap<>();
	private final Map<Character, List<String>> firstCharWordMap = new HashMap<>();

	private final Set<Integer> uniqueCharInts = new HashSet<>(); // for accumulation during pre-processing
	private final List<String> uniqueCharList = new ArrayList<>(); // for model labels, inference, requires String

	// key = char (int), val = char sorted order index from 0-n for all characters in training set
	// map/encoded to eliminate gaps for efficient 1-hot encoding and to reduce unusued inputs/outputs and associated
	// paramaters
	private final Map<Character, Integer> charMap = new HashMap<>();

	private final List<String> allTitles = new ArrayList<>();
	private final List<String> allTitleNgrams = new ArrayList<>();

	private final Map<String, List<String>> dataSets;

	public DataSwizzler(int min1gramWordLength, int minNgramWords, int maxNgramWords) {
		this.min1gramWordLength = min1gramWordLength;
		this.minNgramWords = minNgramWords;
		this.maxNgramWords = maxNgramWords;

		// key=train|val|test val=List<Pair<class,ngrams>>
		dataSets = new HashMap<>();
		dataSets.put("allTitles", allTitles);
		dataSets.put("allTitleNgrams", allTitleNgrams);
	}

	public static void main(String[] args) throws FileNotFoundException, IOException {
		int min1gramWordLength = 3;
		int minNgramWords = 1;
		int maxNgramWords = 3;
		DataSwizzler ds = new DataSwizzler(min1gramWordLength, minNgramWords, maxNgramWords);
		ds.loadData();
	}

	private void saveMaxNgramLength(BufferedWriter nglw) throws IOException {
		nglw.write(String.format("%d", examplesMaxNgramCharLength));
	}

	private void saveMaxLineLength(BufferedWriter llw) throws IOException {
		llw.write(String.format("%d", examplesMaxCharLength));
	}

	private void accumulateUniqueChars(String sequence) {
		sequence.chars().forEach((c) -> uniqueCharInts.add(c));
	}

	private void saveCharMap(BufferedWriter cmw) throws IOException {
		Integer[] uniqueChars = uniqueCharInts.toArray(new Integer[0]);
		Arrays.sort(uniqueChars);

		int[] i = { 0 };

		Arrays.asList(uniqueChars).forEach((c) -> {
			uniqueCharList.add(Character.toString((char) c.intValue()));
			charMap.put((char) c.intValue(), i[0]);
			persistChar(i[0]++, c, cmw);
		});
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
				if (examplesMaxNgramCharLength < ngram.length()) {
					examplesMaxNgramCharLength = ngram.length();
				}
				allTitleNgrams.add(ngram);
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

	public void loadData() throws FileNotFoundException, IOException {
		try (BufferedReader r = new BufferedReader(new FileReader(new File(inputFilename))); //
				BufferedWriter lw = new BufferedWriter(new FileWriter(new File(lineOutputFilename))); //
				BufferedWriter ngw = new BufferedWriter(new FileWriter(new File(ngramOutputFilename))); //
				BufferedWriter wmw = new BufferedWriter(new FileWriter(new File(wordStatsOutputFilename))); //
				BufferedWriter fcwmw = new BufferedWriter(new FileWriter(new File(firstCharWordMapOutputFilename))); //
				BufferedWriter cmw = new BufferedWriter(new FileWriter(new File(charMapOutputFilename))); //
				BufferedWriter llw = new BufferedWriter(new FileWriter(new File(maxLineCharLengthOutputFilename))); //
				BufferedWriter nglw = new BufferedWriter(new FileWriter(new File(maxNgramCharLengthOutputFilename))); //
		) {
			String line;
			while ((line = r.readLine()) != null) {
				if (line.trim().isEmpty()) {
					continue;
				}
				line = lineTransforms(line);
				line = charTransforms(line);
				if (examplesMaxCharLength < line.length()) {
					examplesMaxCharLength = line.length();
				}
				allTitles.add(line);
				writeLineOutput(line, lw);
				accumulateUniqueChars(line);
				wordProcessing(line, ngw);
			}
			saveMaxLineLength(llw);
			saveMaxNgramLength(nglw);
			saveWordMaps(wmw, fcwmw);
			saveCharMap(cmw);
			System.out.println("done");
		}
	}

	public int getMaxCharLength() {
		return examplesMaxCharLength;
	}

	public List<String> getDataSet(String string) {
		return dataSets.get(string);
	}

	// pre-process remove numbers and punctuation, toLowerCase, trim
	public List<String> getTransformedInput(String rawExample) {
		// TODO Auto-generated method stub
		return null;
	}

	public int getInputCharCnt() {
		return uniqueCharList.size();
	}

	public List<String> getOutputChars() {
		return uniqueCharList;
	}

	public Map<Character, Integer> getCharMap() {
		return charMap;
	}

	public Character[] getOutputCharsArray() {
		Character[] outputChars = new Character[uniqueCharList.size()];
		int i[] = { 0 };
		uniqueCharList.forEach(c -> {
			outputChars[i[0]] = c.charAt(0);
			i[0] += 1;
		});
		return null;
	}
}
