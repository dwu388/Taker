# Fomo local recorder

This Android app is an offline-first replacement for the logging portion of the
Tasker -> HTTP -> Google Sheets workflow. It listens directly for notifications
from `family.fomo.app` and appends every notification whose selected text contains
the whole word `bought` or `sold` to a local SQLite database.

It does not place trades, click Fomo, require Windows, or require a network
connection. There is deliberately no Internet permission in the manifest.

## Capture behavior

For each Fomo notification the listener:

1. rejects every package except `family.fomo.app`;
2. reads title, regular text, expanded text and text lines;
3. uses expanded text when available, otherwise regular text, otherwise text lines;
4. requires a whole-word `bought` or `sold` match in the selected text;
5. records the event immediately to local SQLite on a single serial writer;
6. keeps every matching callback as its own row; it does not de-duplicate or discard updates.

The row keeps both the raw Android fields and normalized convenience fields:

- Android notification key, ID, tag, group key, channel and category
- Android `postTime` and local capture time
- title, regular text, expanded text and selected text
- bought/sold action
- parsed trader, token, market cap and source amount when recognizable
- whether the live notification had a `contentIntent`
- number of notification action buttons

The `PendingIntent` itself is intentionally not serialized. Android owns that
live object; the notification key is the durable identity to keep if later code
needs to look the active notification up again.

## Database

Database name: `fomo_buy_sell.db`

Table: `fomo_trade_events`

The table is append-only in normal operation. There is no unique constraint on
notification key because Fomo may update the same Android notification and the
recorder's job is to preserve the complete local event stream.

## Run in an emulator

Open this directory in Android Studio and run the `app` configuration on the
emulator that has Fomo installed and signed in.

Then enable:

Settings -> Notifications -> Special app access -> Notification access -> Fomo Recorder

You can also press **Open notification access** from the app dashboard.

Once notification access is enabled, leave the recorder installed. The listener
does not need the dashboard Activity to remain open.

## Check and export

The dashboard shows total rows, buy rows, sell rows and the latest event.

Press **Export all rows to CSV** to write:

`/sdcard/Android/data/com.dwu.fomorecorder/files/Documents/fomo_buy_sell_events.csv`

Then from Windows:

```bat
adb pull /sdcard/Android/data/com.dwu.fomorecorder/files/Documents/fomo_buy_sell_events.csv .
```

The authoritative local store remains SQLite in the app's private database
directory; CSV is only an explicit snapshot for analysis.

## Intentional differences from the old Tasker logger

- no Apps Script or Google Sheets dependency in the capture path;
- Android's original notification `postTime` is stored, not only interception time;
- notification key/ID/tag are retained;
- all qualifying buys and sells are saved locally, independent of later selection,
  sizing, model confidence, market-cap thresholds or source-buy thresholds;
- thesis-only and unrelated Fomo notifications are ignored rather than recorded;
- failed parsing of optional trader/token/amount fields does not discard a valid
  bought/sold row. Only the bought/sold action is required.

This recorder is intended to become the canonical raw input for later selection and
execution logic. Keep filtering and trading decisions downstream so the raw local
history remains complete.
