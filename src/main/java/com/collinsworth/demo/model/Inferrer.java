package com.collinsworth.demo.model;

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

import com.collinsworth.demo.Service;

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

	public List<String> infer(String seedString) {
		// TODO strip punctuation
		String groomedSeed = seedString.toLowerCase();

		long start = System.nanoTime();

		rnn.net.rnnClearPreviousState();

		List<String> suggestions = new ArrayList<>();

		int seedCharIdx = charMap.get(groomedSeed.charAt(0));

		// TODO seed with multiple characters
		suggestions.add(getSample(seedString, 100));

		// System.out.println(charMap + " " + seedCharIdx + " " + charArray[seedCharIdx]);

		long timeNs = System.nanoTime() - start;
		// System.out.println(timeNs);
		float timeMs = ((float) timeNs) / 1000000;

		return suggestions;
	}

	public void randomlySample(int samples, int length) throws NoSuchAlgorithmException {
		for (int i = 0; i < samples; i++) {
			int seedCharIdx = Math.round(rnd.nextFloat() * (charArray.length - 1));
			System.out.println(String.format("%d %c %s", i, charArray[seedCharIdx],
					getSample(charArray[seedCharIdx].toString(), length)));
			rnn.net.rnnClearPreviousState();
		}
	}

	// TODO can be vector of samples for parallelization and more efficiency
	private String getSample(String seedString, int length) {
		long start = System.nanoTime();

		StringBuilder sb = new StringBuilder();
		sb.append(seedString);

		// final for lambda
		final Character[] nextChar = new Character[1];

		// System.out.println(seedString + " " + seedString.length());
		// TODO seed RNN
		seedString.chars().forEach((c) -> {
			INDArray inputFeature = Nd4j.create(new int[] { 1, charMap.size(), 1 }, 'f');
			// System.out.println(charMap);
			// System.out.println(c + " " + charMap.get((char) c));

				inputFeature.putScalar(new int[] { 1, charMap.get((char) c), 1 }, 1);

				// sample
				INDArray output = rnn.net.rnnTimeStep(inputFeature);

				INDArrayIndex[] indices = new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all() };
				INDArray distribution = output.get(indices);

				// TODO no need to sort for production as this slows it down unnecessarily
				List<CharProb> samples = getTopSamples(distribution, sb);
				nextChar[0] = samples.get(0).c;
			});

		// at last seed character append next predicted char before additional inference sampling
		// TODO randomly sample for variation in production, but need to validate and discard garbage (non-words)
		// int charIdx = sampleFromDistribution(distribution);
		// nextChar = charArray[charIdx];
		sb.append(nextChar[0]);

		// sample loop the RNN
		for (int i = 1; i < length - seedString.length(); i++) {
			INDArray inputFeature = Nd4j.create(new int[] { 1, charMap.size(), 1 }, 'f');
			inputFeature.putScalar(new int[] { 1, charMap.get(nextChar[0]), 1 }, 1);

			// sample
			INDArray output = rnn.net.rnnTimeStep(inputFeature);

			INDArrayIndex[] indices = new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all() };
			INDArray distribution = output.get(indices);

			// TODO no need to sort for production as this slows it down unnecessarily
			List<CharProb> samples = getTopSamples(distribution, sb);
			nextChar[0] = samples.get(0).c;
			// TODO randomly sample for variation in production, but need to validate and discard garbage (non-words)
			// int charIdx = sampleFromDistribution(distribution);
			// nextChar = charArray[charIdx];
			sb.append(nextChar[0]);
		}

		long timeNs = System.nanoTime() - start;
		float timeMs = ((float) timeNs) / 1000000;

		return sb.toString();
	}

	private List<CharProb> getTopSamples(INDArray distribution, StringBuilder sb) {
		List<CharProb> probs = new ArrayList<>(distribution.length());
		for (int i = 0; i < distribution.length(); i++) {
			probs.add(new CharProb(charArray[i], distribution.getFloat(i)));
		}
		Collections.sort(probs, (a, b) -> Float.compare(b.prob, a.prob)); // reverse highest to lowest

		// *************** prints individual char probabilities
		System.out.println(probs);
		// System.out.println(probs.get(0));
		// *************** prints individual char probabilities
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
		// invalid probability distribution should sum to 1.0
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
