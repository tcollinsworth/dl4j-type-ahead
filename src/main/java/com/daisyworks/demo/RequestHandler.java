package com.daisyworks.demo;

import io.vertx.core.json.JsonObject;
import io.vertx.ext.web.RoutingContext;

/**
 * @author troy
 *
 */
public abstract class RequestHandler {
	protected Service service;
	protected RoutingContext rc;
	protected JsonObject bodyJson;

	public RequestHandler(RoutingContext rc, Service service) {
		this.service = service;
		this.rc = rc;
		bodyJson = rc.getBodyAsJson();
		handle();
	}

	public abstract void handle();
}