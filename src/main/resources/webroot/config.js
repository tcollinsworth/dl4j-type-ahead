const Config = (function() {

  const schema = window.location.protocol
  let host = 'localhost:8080'
  let port = ''
  let urlPrefix = 'http://localhost:8080/' //if schema is file:

  if (schema !== 'file:') {
    host = window.location.hostname || 'localhost'
    //localhost and port use them
    //if localhost and no port use 3000
    if (window.location.hostname === 'localhost') {
      port = window.location.port || '8080'
    } else {
      //if not localhost use port if provided otherwise no port
      port = window.location.port || ''
    }

    host = host + (port.length === 0 ? '' : ':' + port)
    urlPrefix = schema + '//' + host + '/'
  }

  function getHost() {
    return host
  }

  function getBaseUrl() {
    return urlPrefix
  }

  function getLangInferenceUrl() {
    return urlPrefix + 'lang-inference'
  }

  function getModelAdminUrl() {
    return urlPrefix + 'model-admin'
  }

  return {
    getBaseUrl,
    getLangInferenceUrl,
    getModelAdminUrl
  }
})()
