package com.daisyworks.demo.model;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author troy
 */
public class RecurrentNeuralNet {
	private final int inputFeatureCnt;
	private final int outputCnt;

	private int iterations;
	private double learningRate;
	private int seed; // initialization seed, keep same for reproducibility
	private double regularizationL2;

	public MultiLayerNetwork net;

	public RecurrentNeuralNet(int iterations, double learningRate, int inputFeatureCnt, int outputCnt, int seed, double regularizationL2) {
		this.inputFeatureCnt = inputFeatureCnt;
		this.outputCnt = outputCnt;

		this.iterations = iterations;
		this.learningRate = learningRate;
		this.seed = seed;
		this.regularizationL2 = regularizationL2;

		initializeNewModel();
	}

	/**
	 * @param iterations
	 * @param learningRate
	 * @param seed
	 */
	public void initializeNewModel(int iterations, float learningRate, int seed) {
		this.iterations = iterations;
		this.learningRate = learningRate;
		this.seed = seed;

		initializeNewModel();
	}

	/**
	 * Create a brand new model.
	 */
	public void initializeNewModel() {
		int hiddenNodes = 200;
		int tbpttLength = 50;

		// https://deeplearning4j.org/features //optimzation algorithms, updaters, hyperparameters, Loss/Objective
		// functions, Activation Functions
		NeuralNetConfiguration.ListBuilder listBuilder = new NeuralNetConfiguration.Builder() //
				.iterations(iterations) //
				.learningRate(learningRate) //
				.seed(seed) //

				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // default
				.weightInit(WeightInit.XAVIER) //
				// .updater(new Nesterovs(0.9)) // auto-reduce learning rate as approaches solution
				// .dropOut(0.1) // reduce ovefitting
				// .useDropConnect(true); // reduce ovefitting

				.updater(new RmsProp(0.95)) //
				// .updater(Updater.ADAM) // Adaptive Momentum - Combines AdaGrad and RmsProp
				// .regularization(true) // reduce ovefitting
				// .l2(regularizationL2) // reduce ovefitting
				// .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // reduce
				// exploding/vanishing gradient issues
				// .gradientNormalizationThreshold(1.0) // 0.5 1.0 // reduce exploding/vanishing gradient issues
				// .trainingWorkspaceMode(WorkspaceMode.SINGLE) // reduce GC overhead
				// .inferenceWorkspaceMode(WorkspaceMode.SINGLE) // reduce GC overhead
				.list() //

				.pretrain(false) //
				.backprop(true) //

				.layer(0, new GravesLSTM.Builder() //
						.nIn(inputFeatureCnt) //
						.nOut(hiddenNodes) //
						.name("Input") //
						.activation(Activation.TANH) //
						.build()) //

				.layer(1, new GravesLSTM.Builder() //
						.nIn(hiddenNodes) //
						.nOut(hiddenNodes) //
						.name("Hidden") //
						.activation(Activation.TANH) //
						.build()) //

				.layer(2, new RnnOutputLayer.Builder() //
						.nIn(hiddenNodes) //
						.nOut(outputCnt) //
						.name("Output") //
						.lossFunction(LossFunctions.LossFunction.MCXENT) //
						.activation(Activation.SOFTMAX) //
						// .weightInit(WeightInit.DISTRIBUTION) //
						.dist(new UniformDistribution(0, 1)) //
						.build()); //

		listBuilder //
				.backpropType(BackpropType.TruncatedBPTT) // limits RNN history to reduce computation overhead
				.tBPTTBackwardLength(tbpttLength) // limits RNN history to reduce computation overhead
				.tBPTTForwardLength(tbpttLength); // limits RNN history to reduce computation overhead

		MultiLayerNetwork net = new MultiLayerNetwork(listBuilder.build());
		net.init();
		this.net = net;

		// Print the number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		System.out.println(String.format("features: %d, classifications: %d", inputFeatureCnt, outputCnt));
	}

	/**
	 * @param filePathName
	 *            i.e., trained_mnist_model.zip
	 * @param saveUpdater
	 *            allows additional training
	 * @throws IOException
	 */
	public void saveModel(String filePathName, boolean saveUpdater) throws IOException {
		File locationToSave = new File(filePathName);
		ModelSerializer.writeModel(net, locationToSave, saveUpdater);
	}

	/**
	 * @param filePathName
	 *            i.e., trained_mnist_model.zip
	 * @param saveUpdater
	 *            allows additional training
	 * @throws IOException
	 */
	public void restoreModel(String filePathName, boolean loadUpdater) throws IOException {
		File locationToSave = new File(filePathName);
		net = ModelSerializer.restoreMultiLayerNetwork(locationToSave, loadUpdater);
	}
}
