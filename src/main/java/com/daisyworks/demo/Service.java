package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.IOException;
import java.security.NoSuchAlgorithmException;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.daisyworks.demo.examples.DataSwizzler;
import com.daisyworks.demo.examples.ExampleCharSeqEncodedVectorDataSetIterator;
import com.daisyworks.demo.model.Evaluator;
import com.daisyworks.demo.model.Inferrer;
import com.daisyworks.demo.model.RecurrentNeuralNet;
import com.daisyworks.demo.model.Trainer;

/**
 * @author troy
 */
public class Service {
	private final int min1gramWordLength = 3;
	private final int minNgramWords = 3; // 1
	private final int maxNgramWords = 50; // 3

	private final int PORT = 8080;

	public final int miniBatchSize = 128;
	private final int seed = 123;
	private final int iterations = 3;
	private final double learningRate = 0.1; // 0.1; // 0.02;
	private final double regularizationL2 = 0.00001;

	public int inputFeatureCnt; // characters
	public int outputCharCnt; // output characters
	public Character[] outputChars;

	// The char length of longest example for truncating/padding
	public int maxExampleLength;

	public DataSwizzler swizzler;

	public DataSetIterator trainDataSetIterator;
	public DataSetIterator validationDataSetIterator;
	public DataSetIterator testDataSetIterator;

	public RecurrentNeuralNet rnn;

	// // infers or predicts classification for input observation features
	public Inferrer inferrer;
	// // trains/fits a neural network model based on input observations and supervised labels
	public Trainer trainer;
	// // evaluates the precision and accuracy of a trained model for test/validation data
	public Evaluator evaluator;

	public static void main(String[] args) throws IOException, NoSuchAlgorithmException {
		Service s = new Service();
		s.main();
	}

	private void main() throws IOException, NoSuchAlgorithmException {
		// for development, also requires staticHandler.setCacheEntryTimeout(1) and browser cache disable
		System.setProperty("vertx.disableFileCaching", "true");

		swizzler = new DataSwizzler(min1gramWordLength, minNgramWords, maxNgramWords);
		swizzler.loadData();

		inputFeatureCnt = swizzler.getInputCharCnt();
		outputCharCnt = swizzler.getOutputChars().size();
		outputChars = swizzler.getOutputCharsArray();

		trainDataSetIterator = new ExampleCharSeqEncodedVectorDataSetIterator( //
				"train", //
				swizzler.getMaxNgramCharLength(), // getMaxLineCharLength || getMaxNgramCharLength
				swizzler.getDataSet("allTitleNgrams"), // allTitles || allTitleNgrams
				swizzler.getOutputChars(), //
				miniBatchSize, //
				swizzler.getCharMap());

		rnn = new RecurrentNeuralNet(iterations, learningRate, inputFeatureCnt, outputCharCnt, seed, regularizationL2);

		inferrer = new Inferrer(rnn, swizzler.getCharMap(), this);
		trainer = new Trainer(rnn);

		// Test sampling
		// rnn.restoreModel("src/main/resources/models/model-iteration-2090-score-1.249226689338684.zip", true);
		// inferrer.randomlySample(20, 100);
		train();
		// evaluator = new Evaluator(rnn, trainDataSetIterator, validationDataSetIterator, testDataSetIterator);

		// evaluator.createAndRegisterEvaluationReporter();

		Vertx vertx = Vertx.vertx();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());
		router.route(HttpMethod.POST, "/typeahead").blockingHandler(routingContext -> new TypeAheadRequestHandler(routingContext, this));
		router.route(HttpMethod.POST, "/model-admin").blockingHandler(routingContext -> new ModelAdminRequestHandler(routingContext, this));
		router.route("/*").handler(StaticHandler.create().setCacheEntryTimeout(1));

		vertx.createHttpServer().requestHandler(router::accept).listen(PORT, res -> {
			if (res.succeeded()) {
				System.out.println("Listening: " + PORT);
			} else {
				System.out.println("Failed to launch server: " + res.cause());
				System.exit(-1);
			}
		});
	}

	public void train() throws IOException {
		trainer.train(trainDataSetIterator, validationDataSetIterator, evaluator, this);
	}

	public Character[] getOutputChars() {
		return outputChars;
	}
}
