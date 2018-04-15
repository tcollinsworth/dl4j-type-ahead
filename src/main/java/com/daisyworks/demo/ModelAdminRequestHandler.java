package com.daisyworks.demo;

import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

import java.io.IOException;

public class ModelAdminRequestHandler extends RequestHandler {

	public ModelAdminRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	@Override
	public void handle() {
		boolean fitModel = bodyJson.getBoolean("fitModel", false);
		boolean saveModel = bodyJson.getBoolean("saveModel", false);
		boolean resetModel = bodyJson.getBoolean("resetModel", false);
		boolean loadModel = bodyJson.getBoolean("loadModel", false);

		String modelFilename = bodyJson.getString("modelFilename", null);

		try {
			if (fitModel) {
				service.train();
				System.out.println("Fitting model: " + modelFilename);
			}

			if (saveModel) {
				if (modelFilename == null || modelFilename.isEmpty()) {
					rc.response().setStatusCode(500).end("no filename");
					return;
				}
				service.rnn.saveModel("src/main/resources/models/" + modelFilename + ".zip", true);
				System.out.println("Saved model: " + modelFilename);
			}

			if (resetModel) {
				service.rnn.initializeNewModel();
				System.out.println("Reset model");
			}

			if (loadModel) {
				if (modelFilename == null || modelFilename.isEmpty()) {
					rc.response().setStatusCode(500).end("no filename");
					return;
				}
				service.rnn.restoreModel("src/main/resources/models/" + modelFilename + ".zip", true);
				System.out.println("Loaded model: " + modelFilename);
			}

			rc.response().end(new JsonObject().encode());
		} catch (IOException e) {
			e.printStackTrace();
			rc.response().setStatusCode(500).end(e.getMessage());
		}
	}
}
