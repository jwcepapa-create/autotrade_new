"""End-to-end checks on synthetic data only; not API entitlement evidence."""
import json
import sys
import pandas as pd
import pytest
import run_backtest
import tiingo_data


def prepare(monkeypatch, tmp_path):
    class FakeClient:
        calls = 0
        rate_limits = []
        def daily(self, ticker, start, end):
            self.calls += 1
            rows = []
            for i, date in enumerate(tiingo_data.sessions(start,end)):
                price = 100 + i * .01
                rows.append({'date':str(date.date()), 'open':price, 'high':price+1,
                             'low':price-1, 'close':price, 'volume':1e7,
                             'adjOpen':price,'adjHigh':price+1,'adjLow':price-1,
                             'adjClose':price,'adjVolume':1e7,'divCash':0,'splitFactor':1})
            return tiingo_data.normalize(rows)
    monkeypatch.setattr(tiingo_data, 'Client', FakeClient)
    cache, output = tmp_path / 'data', tmp_path / 'results'
    _, report = tiingo_data.collect('2016-01-01','2017-01-06',cache,output)
    assert report['status'] == 'validated'
    monkeypatch.setattr(sys, 'argv', ['run_backtest.py','--cache',str(cache),'--output',str(output)])
    return cache, output


def test_full_synthetic_pipeline_and_report(monkeypatch, tmp_path):
    _, output = prepare(monkeypatch, tmp_path)
    run_backtest.main()
    summary = json.loads((output/'summary.json').read_text())
    assert summary['metrics']['swing']['trades'] == 0
    assert summary['metrics']['qqq_buy_hold']['trades'] == 1
    assert summary['intraday']['metrics'] is None
    assert summary['intraday']['status'].startswith('skipped')
    assert (output/'report.html').read_text().startswith('<!doctype html>')
    assert len(pd.read_csv(output/'swing_equity.csv')) > 250


def test_changed_data_cannot_produce_stale_success(monkeypatch, tmp_path):
    cache, output = prepare(monkeypatch, tmp_path)
    (output/'summary.json').write_text('old success')
    (output/'user-notes.txt').write_text('keep me')
    with (cache/'QQQ.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(SystemExit) as exc:
        run_backtest.main()
    assert exc.value.code == 1
    assert not (output/'summary.json').exists()
    assert (output/'user-notes.txt').exists()


def test_failed_collection_removes_old_performance(monkeypatch, tmp_path):
    _, output = prepare(monkeypatch, tmp_path)
    (output/'summary.json').write_text('old success')
    def fail():
        raise tiingo_data.DataError('TIINGO_API_KEY is missing')
    monkeypatch.setattr(tiingo_data,'Client',fail)
    _, report = tiingo_data.collect('2016-01-01','2017-01-06',tmp_path/'missing',output)
    assert report['status'] == 'failed'
    assert not (output/'summary.json').exists()
