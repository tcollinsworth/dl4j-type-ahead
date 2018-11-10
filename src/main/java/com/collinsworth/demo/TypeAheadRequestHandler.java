package com.collinsworth.demo;

import io.vertx.core.json.JsonArray;
import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

import java.util.List;

/**
 * @author troy
 *
 */
public class TypeAheadRequestHandler extends RequestHandler {

	public TypeAheadRequestHandler(RoutingContext rc, Service service) {
		super(rc, service);
	}

	@Override
	public void handle() {
		JsonObject respObj = getSuggestions(service, bodyJson);
		rc.response().end(respObj.encode());
	}

	// FIXME return n-type-ahead suggestions
	private JsonObject getSuggestions(Service service, JsonObject bodyJson) {
		String rawExample = bodyJson.getString("text");

		List<String> suggestions = service.inferrer.infer(rawExample);

		JsonObject respObj = new JsonObject();
		respObj.put("suggestions", new JsonArray(suggestions));

		System.out.println(respObj);

		return respObj;
	}
}