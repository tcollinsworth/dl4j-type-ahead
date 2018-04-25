package com.daisyworks.demo.model;

import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.daisyworks.demo.Service;

/**
 * @author troy
 *
 */
public class Inferrer {
	private final Random rnd = new Random();
	private final Service service;
	private final RecurrentNeuralNet rnn;
	private final Map<Character, Integer> charMap;
	private final Character[] charArray;

	public Inferrer(RecurrentNeuralNet rnn, Map<Character, Integer> charMap, Service service) {
		this.service = service;
		this.rnn = rnn;
		this.charMap = charMap;
		this.charArray = service.swizzler.getOutputCharsArray();
	}

	public Output infer(String rawExample) {
		List<String> words = service.swizzler.getTransformedInput(rawExample);

		final StringBuilder sb = new StringBuilder();
		String[] delimiter = { "" };
		words.forEach((word) -> {
			sb.append(delimiter[0]).append(word);
			delimiter[0] = " ";
		});

		String example = sb.toString().trim().toLowerCase();

		// 1-Hot vector

		// encode/scale, remove untrained characters,
		List<Integer> charsEncoded = new ArrayList<>();
		example.chars().forEachOrdered((c) -> {
			Integer charEncoding = charMap.get((char) c);
			if (charEncoding != null) {
				charsEncoded.add(charEncoding);
			}
		});

		// vectorize inputs, create input mask
		// input Dimensions [miniBatchSize,inputSize,inputTimeSeriesLength]
		INDArray input = Nd4j.zeros(new int[] { 1, service.swizzler.getInputCharCnt(), charsEncoded.size() }, 'f');
		for (int i = 0; i < charsEncoded.size(); i++) {
			input.putScalar(new int[] { 0, charsEncoded.get(i), i }, 1.0f);
		}

		// System.out.println("infer inputs: " + inputs.toString());

		long start = System.nanoTime();

		rnn.net.rnnClearPreviousState();
		// Output dimensions [miniBatchSize,outputSize] or 1 x 7 languages
		INDArray outputs = rnn.net.rnnTimeStep(input);
		// System.out.println(outputs);

		// System.out.println(outputs.length);
		// System.out.println(Arrays.toString(outputs));
		// System.out.println(nn.net.summary());
		long timeNs = System.nanoTime() - start;
		// System.out.println(timeNs);
		float timeMs = ((float) timeNs) / 1000000;

		return getOutput(outputs, charsEncoded.size(), timeMs);
	}

	private Output getOutput(INDArray outputs, int inputTimeSeriesLength, float timeMs) {
		// System.out.println(outputs.shapeInfoToString());
		INDArrayIndex[] lastProbIndices = new INDArrayIndex[] { //
		NDArrayIndex.all(), NDArrayIndex.all(), //
				NDArrayIndex.point(inputTimeSeriesLength - 1) };
		INDArray lastClassificationProbabilities = outputs.get(lastProbIndices);
		int classificationIdx = -1;
		double maxProbability = 0.0;
		List<String> lastProbabilities = new ArrayList<>();
		for (int i = 0; i < service.getOutputChars().length; i++) {
			lastProbabilities.add(service.getOutputChars()[i] + ":" + lastClassificationProbabilities.getFloat(i));
			if (lastClassificationProbabilities.getDouble(i) > maxProbability) {
				maxProbability = lastClassificationProbabilities.getDouble(i);
				classificationIdx = i;
			}
		}
		return new Output(classificationIdx, lastProbabilities, timeMs, outputs.toString());
	}

	public static class Output {
		public final int classificationIdx;
		public final List<String> classificationProbabilities;
		public final float timeMs;
		public final String probMatrix;

		public Output(int classificationIdx, List<String> classificationProbabilities, float timeMs, String probMatrix) {
			this.classificationIdx = classificationIdx;
			this.classificationProbabilities = classificationProbabilities;
			this.timeMs = timeMs;
			this.probMatrix = probMatrix;
		}
	}

	public void randomlySample(int samples, int length) throws NoSuchAlgorithmException {
		for (int i = 0; i < samples; i++) {
			int seedCharIdx = Math.round(rnd.nextFloat() * (charArray.length - 1));
			System.out.println(String.format("%d %c %s", i, charArray[seedCharIdx], getSample(charArray[seedCharIdx], length)));
			rnn.net.rnnClearPreviousState();
		}
	}

	// TODO can be vector of samples for parallelization and more efficiency
	private String getSample(Character seedChar, int length) {
		StringBuilder sb = new StringBuilder();
		sb.append(seedChar);

		Character nextChar = seedChar;
		for (int i = 1; i < length; i++) {
			INDArray inputFeature = Nd4j.create(new int[] { 1, charMap.size(), 1 }, 'f');
			inputFeature.putScalar(new int[] { 1, charMap.get(nextChar), 1 }, 1);

			// sample
			INDArray output = rnn.net.rnnTimeStep(inputFeature);

			INDArrayIndex[] indices = new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all() };
			INDArray distribution = output.get(indices);

			List<CharProb> samples = getTopSamples(distribution, sb);
			nextChar = samples.get(0).c;
			// int charIdx = sampleFromDistribution(distribution);
			// nextChar = charArray[charIdx];
			sb.append(nextChar);
		}

		return sb.toString();
	}

	private List<CharProb> getTopSamples(INDArray distribution, StringBuilder sb) {
		List<CharProb> probs = new ArrayList<>(distribution.length());
		for (int i = 0; i < distribution.length(); i++) {
			probs.add(new CharProb(charArray[i], distribution.getFloat(i)));
		}
		Collections.sort(probs, (a, b) -> Float.compare(b.prob, a.prob)); // reverse highest to lowest

		System.out.println(probs);
		// System.out.println(probs.get(0));
		return probs;
	}

	public int sampleFromDistribution(INDArray distribution) {
		double d = rnd.nextDouble();
		double sum = 0.0;
		for (int i = 0; i < distribution.length(); i++) {
			sum += distribution.getDouble(i);
			if (d <= sum) {
				return i;
			}
		}
		// invalid probability distributionm should sum to 1.0
		throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
	}

	public static class CharProb {
		public final Character c;
		public final float prob;

		public CharProb(Character c, Float prob) {
			this.c = c;
			this.prob = prob;
		}

		@Override
		public int hashCode() {
			return c.hashCode();
		}

		@Override
		public boolean equals(Object other) {
			return c.equals(other);
		}

		@Override
		public String toString() {
			return "CharProb [c=" + c + ", prob=" + prob + "]";
		}
	}
}
