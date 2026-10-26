import io
from urllib.error import HTTPError, URLError
import pytest
import pandas as pd
from tiingo_data import Client, DataError, normalize, validate_coverage, sessions, load_daily


def payload(date='2016-01-04', factor=1):
    return dict(date=date, open=10, high=12, low=9, close=11, volume=100,
                adjOpen=10*factor, adjHigh=12*factor, adjLow=9*factor,
                adjClose=11*factor, adjVolume=100/factor, divCash=0, splitFactor=1)


def test_calendar_and_missing_sessions():
    assert sessions('2016-01-01', '2016-01-05').strftime('%Y-%m-%d').tolist() == ['2016-01-04', '2016-01-05']
    d = normalize([payload()])
    with pytest.raises(DataError, match='missing sessions'):
        validate_coverage(d, '2016-01-01', '2016-01-05')
    assert validate_coverage(d, '2016-01-01', '2016-01-04')['bars'] == 1


@pytest.mark.parametrize('code', [401, 403, 404, 429, 500])
def test_errors_never_expose_token_or_body(code):
    calls = []
    def fail(req, timeout):
        calls.append(req)
        assert 'sentinel-secret' not in req.full_url
        assert req.get_header('Authorization') == 'Token sentinel-secret'
        raise HTTPError(req.full_url, code, 'sentinel-secret', {}, io.BytesIO(b'sentinel-secret'))
    client = Client('sentinel-secret', opener=fail, sleeper=lambda _: None)
    with pytest.raises(DataError) as exc:
        client.daily('QQQ', '2016-01-01', '2016-01-05')
    assert 'sentinel-secret' not in str(exc.value)
    assert len(calls) == (3 if code in (429, 500) else 1)


def test_adjustment_and_duplicates():
    d = normalize([payload(factor=.25)])
    assert d.adjClose.iloc[0] == 2.75
    with pytest.raises(DataError, match='duplicate'):
        normalize([payload(), payload()])
    bad = payload(); bad['adjOpen'] = 10.5
    with pytest.raises(DataError, match='adjusted OHLC'):
        normalize([bad])


def test_cache_avoids_network_and_refreshes_adjustments(tmp_path):
    class Fake:
        calls = 0
        def daily(self, ticker, start, end):
            self.calls += 1
            dates = sessions(start, end)
            return normalize([payload(str(x.date()), factor=.5 if self.calls > 1 else 1) for x in dates])
    client = Fake()
    load_daily(client, 'QQQ', '2016-01-01', '2016-01-05', tmp_path)
    load_daily(client, 'QQQ', '2016-01-01', '2016-01-05', tmp_path)
    assert client.calls == 1
    d, info = load_daily(client, 'QQQ', '2016-01-01', '2016-01-06', tmp_path)
    assert client.calls == 3
    assert info['cache_mode'].startswith('full_refresh')
    assert (d.adjClose == 5.5).all()
