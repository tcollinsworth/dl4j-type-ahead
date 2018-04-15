package com.daisyworks.demo;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

import com.daisyworks.demo.model.Inferrer.Output;

/**
 * @author troy
 *
 */
public class LangRequestHandler extends RequestHandler {

	public LangRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	@Override
	public void handle() {
		JsonObject respObj = getLangInference(service, bodyJson);
		rc.response().end(respObj.encode());
	}

	private JsonObject getLangInference(Service service, JsonObject bodyJson) {
		String rawExample = bodyJson.getString("text");

		Output output = service.inferrer.infer(rawExample);

		String lang = service.getClassifications()[output.classificationIdx];

		JsonObject respObj = new JsonObject();
		respObj.put("lang", lang);

		JsonArray classProbabilities = new JsonArray(output.classificationProbabilities);
		respObj.put("langProbabilities", classProbabilities);
		respObj.put("timeMs", output.timeMs);
		respObj.put("probMatrix", output.probMatrix);
		System.out.println(output.probMatrix);
		System.out.println(respObj);

		return respObj;
	}
}