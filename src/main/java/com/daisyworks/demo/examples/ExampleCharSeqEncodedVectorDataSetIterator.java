package com.daisyworks.demo.examples;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ExampleCharSeqEncodedVectorDataSetIterator implements DataSetIterator {
	private static final long serialVersionUID = 1L;
	private static final Logger Log = LoggerFactory.getLogger(ExampleCharSeqEncodedVectorDataSetIterator.class);

	private final String dataSetName;
	private final List<String> dataSet;

	private final List<String> outputChars;

	private final Map<Character, Integer> charMap;

	// The char length of longest example for truncating/padding
	private final int exampleMaxCharLength;

	private final int miniBatchSize;

	private int numExamples = -1;

	private int cursor = 0;

	public ExampleCharSeqEncodedVectorDataSetIterator( //
			String dataSetName, //
			int exampleMaxCharLength, //
			List<String> dataSet, //
			List<String> outpuChars, //
			int miniBatchSize, //
			Map<Character, Integer> charMap) {
		this.dataSetName = dataSetName;
		this.dataSet = dataSet;
		this.exampleMaxCharLength = exampleMaxCharLength;
		this.outputChars = outpuChars;
		this.miniBatchSize = miniBatchSize;
		this.charMap = charMap;
	}

	/**
	 * @param num
	 *            mini-batch size
	 * @return
	 */
	@Override
	public DataSet next(int num) {
		List<String> miniBatch = getMiniBatchExamples(num);
		Log.debug("next {} cursor {} miniBatchSize {}", num, cursor, miniBatch.size());
		if (cursor == numExamples - 1) {
			return null;
		}
		DataSet ds = getDataSet(miniBatch);
		return ds;
	}

	/**
	 * <pre>
	 * 
	 *     T  examples
	 * 	  O  +--+--+--+
	 *   H  /1 /  /  /|
	 * 	/  +--+--+--+ +
	 * 1  /  /  /  /|/|
	 * 	 +--+0-+--+ + +
	 *   |  |  |  |/|/
	 *   +--+--+--+ +   
	 *   |2 |  |  |/
	 *   +--+--+--+
	 *    e  e  e
	 *    x  x  x
	 *    a  a  a
	 *    m  m  m
	 *    p  p  p
	 *    l  l  l
	 *    e  e  e
	 *    
	 *    0  1  2
	 * 
	 * 	 dimension 0 = minibatch size or examples
	 * 	 dimension 1 = uniqueChars (1-hot) vector - each input represents unique character, only 1 input is high/1 in vector
	 * 	 dimension 2 = max minibatch example length (char sequence length) - holds a single training or inference example
	 * 	 vector memory ordering c = row first, order f (fortran) = column first - use f for most efficiency
	 * https://nd4j.org/userguide.html?__hstc=3042607.d889426ce588e6f9166b006d9576c0e8.1523754595153.1523754595153.1523820923997.2&__hssc=3042607.1.1523820923997&__hsfp=2102497138#inmemory
	 * </pre>
	 */
	private DataSet getDataSet(List<String> miniBatchExamples) {
		INDArray inputFeatureMatrix = Nd4j.create(new int[] { miniBatchExamples.size(), charMap.size(), exampleMaxCharLength }, 'f');
		// Log.info("inputFeatureMatrix {}", inputFeatureMatrix.shapeInfoToString());
		INDArray labelsMatrix = Nd4j.create(new int[] { miniBatchExamples.size(), outputChars.size(), exampleMaxCharLength }, 'f');
		// Log.info("labelsMatrix {}", labelsMatrix.shapeInfoToString());

		// Don't need masks because every input has output prediction label
		// Where's the input features located? Masks 1 if input feature present, 0 for padding
		// INDArray featuresMaskMatrix = Nd4j.zeros(miniBatchExamples.size(), exampleMaxCharLength);
		// Log.info("featuresMaskMatrix {}", featuresMaskMatrix.shapeInfoToString());
		// Where's the output labels located? Masks 1 if label present, 0 for padding
		// INDArray labelsMaskMatrix = Nd4j.zeros(miniBatchExamples.size(), exampleMaxCharLength);
		// Log.info("labelsMaskMatrix {}", labelsMaskMatrix.shapeInfoToString());

		// iterate examples, then example setting 1-HOT
		for (int mbIdx = 0; mbIdx < miniBatchExamples.size(); mbIdx++) {
			String example = miniBatchExamples.get(mbIdx);
			// since next character in sequence is the prediction, the last example character has no output prediction
			// label
			for (int exampleIdx = 0; exampleIdx < example.length() - 1; exampleIdx++) {
				int oneHotCharIdx = charMap.get(example.charAt(exampleIdx));
				inputFeatureMatrix.putScalar(new int[] { mbIdx, oneHotCharIdx, exampleIdx }, 1);

				int labelOneHotCharIdx = charMap.get(example.charAt(exampleIdx + 1));
				labelsMatrix.putScalar(new int[] { mbIdx, labelOneHotCharIdx, exampleIdx }, 1);

				// featuresMaskMatrix.putScalar(exampleIdx, 1);
				// labelsMaskMatrix.putScalar(exampleIdx, 1);
			}
		}

		DataSet ds = new DataSet(inputFeatureMatrix, labelsMatrix);
		Log.debug("inputFeatureMatrix {}", inputFeatureMatrix.shapeInfoToString());
		Log.debug("labelsMatrix {}", labelsMatrix.shapeInfoToString());
		// Log.debug("featuresMaskMatrix {}", featuresMaskMatrix.shapeInfoToString());
		// Log.debug("labelsMaskMatrix {}", labelsMaskMatrix.shapeInfoToString());
		return ds;
	}

	private List<String> getMiniBatchExamples(int num) {
		int endCursor = cursor + num;
		List<String> miniBatch = new ArrayList<>();
		for (; cursor < dataSet.size() - 1 && cursor < endCursor; cursor++) {
			miniBatch.add(dataSet.get(cursor));
		}
		return miniBatch;
	}

	@Override
	public boolean hasNext() {
		return cursor + 1 < numExamples();
	}

	@Override
	public DataSet next() {
		return next(miniBatchSize);
	}

	@Override
	public int totalExamples() {
		if (numExamples != -1) {
			return numExamples;
		}
		return numExamples = dataSet.size();
	}

	@Override
	public int inputColumns() {
		return 1;
	}

	@Override
	public int totalOutcomes() {
		return outputChars.size();
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		cursor = 0;
		Collections.shuffle(dataSet);
	}

	@Override
	public int batch() {
		return miniBatchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return outputChars;
	}

	@Override
	public String toString() {
		return "ExampleCharSeqEncodedVectorDataSetIterator [dataSetName=" + dataSetName + ", exampleMaxCharLength=" + exampleMaxCharLength + ", miniBatchSize=" + miniBatchSize
				+ ", numExamples=" + numExamples + ", cursor=" + cursor + "]";
	}
}
