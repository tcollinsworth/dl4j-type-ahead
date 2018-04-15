package com.daisyworks.demo;

import io.vertx.core.Vertx;
import io.vertx.core.http.HttpMethod;
import io.vertx.ext.web.Router;
import io.vertx.ext.web.handler.BodyHandler;
import io.vertx.ext.web.handler.StaticHandler;

import java.io.IOException;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.daisyworks.demo.language.DataSwizzler;
import com.daisyworks.demo.model.Evaluator;
import com.daisyworks.demo.model.Inferrer;
import com.daisyworks.demo.model.RecurrentNeuralNet;
import com.daisyworks.demo.model.Trainer;

/**
 * @author troy
 */
public class Service {
	private final int min1gramWordLength = 3;
	private final int minNgramWords = 1;
	private final int maxNgramWords = 3;

	private final int PORT = 8080;

	public final int miniBatchSize = 32;
	private final int seed = 123;
	private final int iterations = 1;
	private final double learningRate = 0.1; // 0.1; // 0.02;
	private final double regularizationL2 = 0.00001;

	public int inputFeatureCnt; // characters
	public int outputCnt; // classifications
	public String[] classifications;

	// The char length of longest example for truncating/padding
	public int maxExampleLength;

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

	public static void main(String[] args) throws IOException {
		Service s = new Service();
		s.main();
	}

	private void main() throws IOException {
		// for development, also requires staticHandler.setCacheEntryTimeout(1) and browser cache disable
		System.setProperty("vertx.disableFileCaching", "true");

		DataSwizzler swizzler = new DataSwizzler(min1gramWordLength, minNgramWords, maxNgramWords);
		swizzler.loadData();

		outputCnt = swizzler.getClassificationSet().size();
		classifications = swizzler.getClassificationSet().toArray(new String[0]);

		trainDataSetIterator = new ExampleCharSeqAsDoubleEncodedVectorDataSetIterator( //
				"train", //
				swizzler.getMaxCharLength(), //
				swizzler.getDataSet("train"), //
				swizzler.getClassificationSet(), //
				miniBatchSize, //
				swizzler.getCharMap());

		rnn = new RecurrentNeuralNet(iterations, learningRate, 1, outputCnt, seed, regularizationL2);

		inferrer = new Inferrer(rnn, swizzler.getCharMap(), this);
		trainer = new Trainer(rnn);
		evaluator = new Evaluator(rnn, trainDataSetIterator, validationDataSetIterator, testDataSetIterator);

		evaluator.createAndRegisterEvaluationReporter();

		Vertx vertx = Vertx.vertx();
		Router router = Router.router(vertx);
		router.route().handler(BodyHandler.create());
		router.route(HttpMethod.POST, "/lang-inference").blockingHandler(routingContext -> new LangRequestHandler(routingContext, this));
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
		trainer.train(trainDataSetIterator, validationDataSetIterator, evaluator);
	}

	public String[] getClassifications() {
		return classifications;
	}
}
