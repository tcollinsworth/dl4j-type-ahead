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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
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

	private DataSet getDataSet(List<String> miniBatchExamples) {

		// FIXME

		INDArray inputFeatureMatrix = Nd4j.create(new int[] { miniBatchExamples.size(), 1, exampleMaxCharLength }, 'f');
		INDArray labelsMatrix = Nd4j.create(new int[] { miniBatchExamples.size(), outputChars.size(), exampleMaxCharLength }, 'f');
		// Masks 1 if data present, 0 for padding
		INDArray featuresMaskMatrix = Nd4j.zeros(miniBatchExamples.size(), exampleMaxCharLength);
		INDArray labelsMaskMatrix = Nd4j.zeros(miniBatchExamples.size(), exampleMaxCharLength);

		for (int exampleIdx = 0; exampleIdx < miniBatchExamples.size(); exampleIdx++) {
			String example = miniBatchExamples.get(exampleIdx);

			INDArrayIndex[] indices = new INDArrayIndex[] { //
			NDArrayIndex.point(exampleIdx), //
					NDArrayIndex.all(), //
					NDArrayIndex.interval(0, exampleMaxCharLength) };

			// inputFeatureMatrix.putRow(exampleIdx, getExampleMatrix(example)); //simpler, same effect/result
			inputFeatureMatrix.put(indices, getExampleMatrix(example));

			// for current example, set each corresponding feature mask value to 1 for the length of the
			// example, leaving padding values 0
			featuresMaskMatrix.get(new INDArrayIndex[] { NDArrayIndex.point(exampleIdx), NDArrayIndex.interval(0, example.length()) }).assign(1);

			int classIdx = 0; // FIXME getLabelClassIdx(classification);
			int labelAtLastFeatureIdx = example.length() - 1;
			labelsMatrix.putScalar(new int[] { exampleIdx, classIdx, labelAtLastFeatureIdx }, 1.0);

			labelsMaskMatrix.putScalar(new int[] { exampleIdx, labelAtLastFeatureIdx }, 1.0);
		}

		DataSet ds = new DataSet(inputFeatureMatrix, labelsMatrix, featuresMaskMatrix, labelsMaskMatrix);
		Log.debug("inputFeatureMatrix {}", inputFeatureMatrix.shapeInfoToString());
		Log.debug("labelsMatrix {}", labelsMatrix.shapeInfoToString());
		Log.debug("featuresMaskMatrix {}", featuresMaskMatrix.shapeInfoToString());
		Log.debug("labelsMaskMatrix {}", labelsMaskMatrix.shapeInfoToString());
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

	private INDArray getExampleMatrix(String example) {
		INDArray exampleMatrix = Nd4j.zeros(exampleMaxCharLength, 1);

		// iterate the example char sequence
		for (int exampleCharIdx = 0; exampleCharIdx < example.length(); exampleCharIdx++) {
			double uniqueCharIdx = charMap.get(example.charAt(exampleCharIdx));
			exampleMatrix.putScalar(new int[] { exampleCharIdx, 0 }, uniqueCharIdx);
		}
		Log.debug("exampleMatrix {}", exampleMatrix.shapeInfoToString());
		return exampleMatrix;
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
