package com.collinsworth.demo.model;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.collinsworth.demo.Service;

/**
 * @author troy
 *
 */
public class Trainer {
	RecurrentNeuralNet rnn;

	public Trainer(RecurrentNeuralNet rnn) {
		this.rnn = rnn;

		// https://deeplearning4j.org/workspaces
		// limit gc frequency - 10000 milliseconds
		// Nd4j.getMemoryManager().setAutoGcWindow(10000);

		// OR disable
		// Nd4j.getMemoryManager().togglePeriodicGc(false);
	}

	public void train(DataSetIterator trainDataSetIterator, DataSetIterator validationDataSetIterator, Evaluator evaluator, Service service) throws IOException {
		// Evaluator.printStatsHeader();
		// evaluator.printStats();

		for (int i = 0; i < 100000; i++) {
			// long start = System.currentTimeMillis();
			// trainDataSetIterator.reset();
			// validationDataSetIterator.reset();
			// testDataSetIterator.reset();

			fit(trainDataSetIterator);

			trainDataSetIterator.reset();
			// validationDataSetIterator.reset();
			// testDataSetIterator.reset();

			// double valAccuracy = evaluator.printStats();
			if (i % 1 == 0) {
				boolean saveUpdater = true;
				rnn.saveModel("src/main/resources/models/model-iteration-" + i + "-score-" + rnn.net.score() + ".zip", saveUpdater);
				try {
					service.inferrer.randomlySample(20, 100);
				} catch (NoSuchAlgorithmException e) {
					e.printStackTrace();
				}
			}
			// System.out.println("interation train eval time " + ((System.currentTimeMillis() - start) / 1000) +
			// " sec");
		}
	}

	int fitCnt = 0;

	public void fit(DataSetIterator trainDataSetIterator) {
		rnn.net.fit(trainDataSetIterator);
	}
}
