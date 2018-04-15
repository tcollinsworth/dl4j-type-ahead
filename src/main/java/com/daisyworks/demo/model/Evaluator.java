package com.daisyworks.demo.model;

import java.util.Date;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.daisyworks.demo.Service;

/**
 * @author troy
 *
 */
public class Evaluator {
	private final RecurrentNeuralNet rnn;

	private final DataSetIterator trainDataSetIterator;
	private final DataSetIterator validationDataSetIterator;
	private final DataSetIterator testDataSetIterator;

	private int lastIteration = -1;
	private double lastIterationScore = -1;

	public Evaluator(RecurrentNeuralNet rnn, DataSetIterator trainDataSetIterator, DataSetIterator validationDataSetIterator, DataSetIterator testDataSetIterator) {
		this.rnn = rnn;

		this.trainDataSetIterator = trainDataSetIterator;
		this.validationDataSetIterator = validationDataSetIterator;
		this.testDataSetIterator = testDataSetIterator;
	}

	public Evaluation grade(Service service, DataSetIterator dataSet) {
		return rnn.net.evaluate(dataSet);
	}

	public void createAndRegisterEvaluationReporter() {
		rnn.net.setListeners(new EvaluationReporter());
	}

	public static void printStatsHeader() {
		System.out.println("date, iteration, score, trainAccuracy, trainF1, trainPrecision, valAccuracy, valF1, valPrecision, testAccuracy, testF1, testPrecision");
	}

	public double printStats() {
		Evaluation trainEval = rnn.net.evaluate(trainDataSetIterator);
		// This is used to tune various model architecture hyperparameters, which could lead to overfitting
		Evaluation valEval = rnn.net.evaluate(validationDataSetIterator);

		if (testDataSetIterator != null) {
			// This is never used to tune models and is only used for inter-model evaluation
			Evaluation testEval = rnn.net.evaluate(testDataSetIterator);
			// TODO print results
		}

		System.out.println(String.format("%s %d %f %f %f %f %f %f %f", new Date().toString(), lastIteration + 1, lastIterationScore, trainEval.accuracy(), trainEval.f1(),
				trainEval.precision(), valEval.accuracy(), valEval.f1(), valEval.precision()));
		return valEval.accuracy();
	}

	class EvaluationReporter implements IterationListener {
		private static final long serialVersionUID = 1L;

		private boolean invoked = false;

		@Override
		public boolean invoked() {
			return invoked;
		}

		@Override
		public void invoke() {
			invoked = true;
		}

		@Override
		public void iterationDone(Model model, int iteration) {
			invoke();
			lastIteration = iteration;
			lastIterationScore = model.score();
			// calling evaluation from here causes java.lang.IllegalStateException: Mis matched shapes
			// printStats(iteration, score);
		}
	}
}
