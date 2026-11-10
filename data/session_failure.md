# Q10: three sessions in a row without progress

The kernel ended with status `HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/kernels.KernelsApiService/GetKernelSessionStatus (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:1002)')))`, but the output still has no step beyond 35100.
Each retry pushed the same session again; three of them came back empty, so this
is not a quota wait -- something is wrong with the run and it needs a look.

## Log tail

```n
The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\adapters.py", line 696, in se
nd
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\urllib3\connectionpool.py", line 847, 
in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\urllib3\util\retry.py", line 555, in i
ncrement
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: 
/v1/security.OAuthService/IntrospectToken (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_s
sl.c:1002)')))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 1038, in _bootstrap_inner
[log] no data (session may not have started)
    self.run()
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\threading.py", line 975, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\Icold\Desktop\kron_llm\scripts\kaggle_log.py", line 51, in worker
    api.authenticate()
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\kaggle\api\kaggle_api_extended.py", li
ne 1496, in authenticate
    if self._authenticate_with_access_token():
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\kaggle\api\kaggle_api_extended.py", li
ne 1567, in _authenticate_with_access_token
    username = self._introspect_token(access_token)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\kaggle\api\kaggle_api_extended.py", li
ne 1609, in _introspect_token
    response = kaggle.security.oauth_client.introspect_token(request)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\kagglesdk\security\services\oauth_serv
ice.py", line 44, in introspect_token
    return self._client.call("security.OAuthService", "IntrospectToken", request, IntrospectTokenResponse)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\kagglesdk\kaggle_http_client.py", line
 104, in call
    http_response = self._session.send(http_request, **settings)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\sessions.py", line 784, in se
nd
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Icold\AppData\Local\Programs\Python\Python311\Lib\site-packages\requests\adapters.py", line 727, in se
nd
    raise SSLError(e, request=request)
requests.exceptions.SSLError: HTTPSConnectionPool(host='api.kaggle.com', port=443): Max retries exceeded with url: /v1/
security.OAuthService/IntrospectToken (Caused by SSLError(SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c
:1002)')))
```
