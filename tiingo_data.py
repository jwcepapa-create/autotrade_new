"""Tiingo EOD client: credentials in Authorization only; never log response bodies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import exchange_calendars as xcals
import numpy as np
import pandas as pd

TICKERS = ('QQQ', 'SPY', 'NVDA', 'TSLA', 'AMD', 'AMZN', 'META')
FIELDS = ['open', 'high', 'low', 'close', 'volume', 'adjOpen', 'adjHigh',
          'adjLow', 'adjClose', 'adjVolume', 'divCash', 'splitFactor']


class DataError(RuntimeError):
    pass


class Client:
    def __init__(self, token=None, opener=urlopen, sleeper=time.sleep):
        self.token = token if token is not None else os.getenv('TIINGO_API_KEY', '')
        if not self.token.strip():
            raise DataError('TIINGO_API_KEY is missing; configure the Actions secret.')
        self.opener, self.sleeper = opener, sleeper
        self.calls = 0
        self.rate_limits = []

    def get(self, path, params):
        url = 'https://api.tiingo.com/' + path + '?' + urlencode(params)
        req = Request(url, headers={'Authorization': 'Token ' + self.token,
                                    'Accept': 'application/json', 'User-Agent': 'research-backtest/1.0'})
        for attempt in range(3):
            try:
                self.calls += 1
                with self.opener(req, timeout=40) as response:
                    # Retain only numeric, non-secret quota headers, never arbitrary headers.
                    quota = {k.lower(): v for k, v in response.headers.items()
                             if k.lower().startswith('x-ratelimit') and str(v).isdigit()}
                    if quota:
                        self.rate_limits.append(quota)
                    payload = json.load(response)
                if not isinstance(payload, list):
                    raise DataError('Unexpected Tiingo response schema (expected a price list).')
                return payload
            except HTTPError as exc:
                status = exc.code
                if status in (429, 500, 502, 503, 504) and attempt < 2:
                    self.sleeper(min(20, 2 ** (attempt + 1)))
                    continue
                reasons = {401: 'invalid or missing API token', 403: 'account/plan permission denied',
                           404: 'ticker or endpoint unavailable', 429: 'API quota/rate limit exceeded'}
                raise DataError(f'Tiingo HTTP {status}: {reasons.get(status, "provider request failed")}.') from None
            except (URLError, TimeoutError, OSError):
                if attempt < 2:
                    self.sleeper(2 ** (attempt + 1))
                    continue
                raise DataError('Tiingo network/TLS/timeout failure after 3 attempts.') from None
            except (ValueError, TypeError):
                raise DataError('Tiingo returned invalid JSON; response body suppressed.') from None

    def daily(self, ticker, start, end):
        if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,14}', ticker):
            raise DataError('Invalid ticker format.')
        return normalize(self.get(f'tiingo/daily/{ticker}/prices',
                                 {'startDate': start, 'endDate': end, 'resampleFreq': 'daily'}))


def normalize(payload):
    if not payload:
        raise DataError('Tiingo returned no bars: check date range and plan coverage.')
    d = pd.DataFrame(payload)
    if not {'date', *FIELDS}.issubset(d.columns):
        raise DataError('Missing raw/adjusted OHLCV or corporate-action fields.')
    try:
        d.index = pd.to_datetime(d.pop('date'), utc=True).dt.tz_localize(None).dt.normalize()
        d = d[FIELDS].apply(pd.to_numeric, errors='raise').sort_index()
    except (ValueError, TypeError):
        raise DataError('Invalid dates or numeric values in Tiingo data.') from None
    d.index.name = 'Date'
    validate_values(d)
    return d


def validate_values(d):
    if d.empty or d.index.has_duplicates or not d.index.is_monotonic_increasing:
        raise DataError('Empty, duplicate or unsorted daily bars.')
    if not set(FIELDS).issubset(d.columns) or not np.isfinite(d[FIELDS].to_numpy()).all():
        raise DataError('Missing/non-finite OHLCV or corporate-action values.')
    for prefix in ('', 'adj'):
        names = [prefix + n if prefix else n.lower() for n in ('Open', 'High', 'Low', 'Close')]
        o, h, lo, c = (d[n] for n in names)
        if ((lo <= 0) | (h < lo) | (h < o) | (h < c) | (lo > o) | (lo > c)).any():
            raise DataError('Invalid OHLC price ordering or non-positive prices.')
    if (d[['volume', 'adjVolume', 'divCash']] < 0).any().any() or (d.splitFactor <= 0).any():
        raise DataError('Invalid volume/dividend/split factor.')
    # Tiingo adjusts all OHLC by the same daily factor, including dividends/splits.
    factor = d.adjClose / d.close
    for raw, adj in [('open', 'adjOpen'), ('high', 'adjHigh'), ('low', 'adjLow')]:
        if not np.allclose(d[adj], d[raw] * factor, rtol=1e-5, atol=1e-6):
            raise DataError('Inconsistent adjusted OHLC factors.')


def sessions(start, end):
    cal = xcals.get_calendar('XNYS', start=pd.Timestamp(start) - pd.Timedelta(days=7),
                             end=pd.Timestamp(end) + pd.Timedelta(days=7))
    return cal.sessions[(cal.sessions >= pd.Timestamp(start)) & (cal.sessions <= pd.Timestamp(end))].tz_localize(None)


def validate_coverage(d, start, end):
    validate_values(d)
    expected = sessions(start, end)
    missing = expected.difference(d.index)
    unexpected = d.loc[start:end].index.difference(expected)
    if len(missing) or len(unexpected):
        examples = ','.join(x.strftime('%Y-%m-%d') for x in missing[:8])
        raise DataError(f'Data coverage failed: {len(missing)} missing sessions ({examples}); '
                        f'{len(unexpected)} unexpected sessions. No forward filling.')
    return {'bars': len(d.loc[start:end]), 'first': str(d.loc[start:end].index[0].date()),
            'last': str(d.loc[start:end].index[-1].date()), 'missing_sessions': 0,
            'split_events': int((d.loc[start:end].splitFactor != 1).sum()),
            'dividend_events': int((d.loc[start:end].divCash > 0).sum())}


def load_daily(client, ticker, start, end, cache_dir, refresh=False):
    """Incremental overlap, full refresh if historical adjustment factors change."""
    path = Path(cache_dir) / f'{ticker}.csv'
    cached = None
    if path.exists() and not refresh:
        try:
            cached = pd.read_csv(path, index_col='Date', parse_dates=True)
            validate_values(cached)
        except (ValueError, OSError, DataError):
            cached = None
    if cached is not None and cached.index.min() <= sessions(start, end)[0]:
        if cached.index.max() >= sessions(start, end)[-1]:
            d, mode = cached, 'cache'
        else:
            overlap_start = max(pd.Timestamp(start), cached.index.max() - pd.Timedelta(days=14))
            new = client.daily(ticker, overlap_start.date().isoformat(), end)
            common = cached.index.intersection(new.index)
            changed = not len(common) or not np.allclose(
                cached.loc[common, FIELDS], new.loc[common, FIELDS], rtol=1e-7, atol=1e-8)
            later = new.loc[new.index > cached.index.max()]
            action = (later.divCash > 0).any() or (later.splitFactor != 1).any()
            if changed or action:
                d, mode = client.daily(ticker, start, end), 'full_refresh_corporate_action_or_revision'
            else:
                d = pd.concat([cached.drop(index=common), new]).sort_index()
                mode = 'incremental'
    else:
        d, mode = client.daily(ticker, start, end), 'full_download'
    info = validate_coverage(d, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    d.to_csv(temp)
    temp.replace(path)
    info['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    info['cache_mode'] = mode
    return d.loc[start:end], info


def adjusted(d):
    return d[['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']].rename(
        columns=dict(zip(['adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume'],
                         ['Open', 'High', 'Low', 'Close', 'Volume'])))


def clear_performance(output):
    # Only our own known products, never unrelated files in a user-selected directory.
    names = ['summary.json', 'metrics.csv', 'report.html', 'swing_signals.csv',
             'swing_trades.csv', 'qqq_buy_hold_trades.csv', 'swing_equity.csv', 'qqq_buy_hold_equity.csv']
    for name in names:
        (Path(output) / name).unlink(missing_ok=True)


def collect(start, end, cache_dir, output, refresh=False, probe_intraday=False):
    clear_performance(output)
    report = {'provider': 'Tiingo', 'start': start, 'end': end, 'tickers': {},
              'intraday': {'status': 'not_verified', 'reason': 'No 5-minute entitlement assumed.'},
              'plan': 'Account tier not exposed by prices API; successful requests prove only requested coverage.'}
    frames = {}
    try:
        client = Client()
        for ticker in TICKERS:
            try:
                raw, info = load_daily(client, ticker, start, end, cache_dir, refresh)
                frames[ticker] = adjusted(raw)
                report['tickers'][ticker] = {'status': 'ok', **info}
                print(f'{ticker}: {info["bars"]} daily bars, {info["first"]}..{info["last"]}; {info["cache_mode"]}')
            except DataError as exc:
                report['tickers'][ticker] = {'status': 'failed', 'reason': str(exc)}
                print(f'{ticker}: {exc}')
        if probe_intraday:
            try:
                bars = client.get('iex/QQQ/prices', {'startDate': '2016-01-04',
                                  'endDate': '2016-01-05', 'resampleFreq': '5min'})
                report['intraday'] = {'status': 'probe_only' if bars else 'unavailable',
                    'bars': len(bars), 'reason': 'Probe is not a complete 2016-present intraday dataset; strategy skipped.'}
            except DataError as exc:
                report['intraday'] = {'status': 'unavailable', 'reason': str(exc)}
        report['http_attempts'] = client.calls
        report['rate_limit_headers'] = client.rate_limits
        report['quota_note'] = 'Serial requests; bounded backoff on 429. Successful calls do not establish remaining account quota.'
    except DataError as exc:
        report['error'] = str(exc)
        print(str(exc))
    report['status'] = 'validated' if len(frames) == len(TICKERS) else 'failed'
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / 'data_validation.json').write_text(json.dumps(report, indent=2))
    return frames, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default=(pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None) - pd.Timedelta(days=1)).date().isoformat())
    ap.add_argument('--cache', default='data')
    ap.add_argument('--output', default='results')
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--probe-intraday', action='store_true')
    args = ap.parse_args()
    _, report = collect(args.start, args.end, args.cache, args.output, args.refresh, args.probe_intraday)
    raise SystemExit(0 if report['status'] == 'validated' else 1)


if __name__ == '__main__':
    main()
