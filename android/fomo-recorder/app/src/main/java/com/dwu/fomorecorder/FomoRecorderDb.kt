package com.dwu.fomorecorder

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.File

data class RecorderStats(
    val total: Long,
    val buys: Long,
    val sells: Long,
    val latest: String?,
)

class FomoRecorderDb(context: Context) :
    SQLiteOpenHelper(context.applicationContext, DB_NAME, null, DB_VERSION) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE fomo_trade_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                package_name TEXT NOT NULL,
                notification_key TEXT NOT NULL,
                notification_id INTEGER NOT NULL,
                notification_tag TEXT,
                post_time_ms INTEGER NOT NULL,
                captured_time_ms INTEGER NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                big_text TEXT,
                selected_text TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('bought', 'sold')),
                trader TEXT,
                token TEXT,
                market_cap_usd REAL,
                source_amount_usd REAL,
                channel_id TEXT,
                category TEXT,
                group_key TEXT,
                has_content_intent INTEGER NOT NULL,
                notification_action_count INTEGER NOT NULL
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_fomo_events_post_time ON fomo_trade_events(post_time_ms)")
        db.execSQL("CREATE INDEX idx_fomo_events_action ON fomo_trade_events(action)")
        db.execSQL("CREATE INDEX idx_fomo_events_trader_token ON fomo_trade_events(trader, token)")
        db.execSQL("CREATE INDEX idx_fomo_events_notification_key ON fomo_trade_events(notification_key)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        error("No database migration exists from $oldVersion to $newVersion")
    }

    fun insert(event: TradeEvent): Long {
        val values = ContentValues().apply {
            put("package_name", event.packageName)
            put("notification_key", event.notificationKey)
            put("notification_id", event.notificationId)
            put("notification_tag", event.notificationTag)
            put("post_time_ms", event.postTimeMs)
            put("captured_time_ms", event.capturedTimeMs)
            put("title", event.title)
            put("text", event.text)
            put("big_text", event.bigText)
            put("selected_text", event.selectedText)
            put("action", event.action)
            put("trader", event.trader)
            put("token", event.token)
            put("market_cap_usd", event.marketCapUsd)
            put("source_amount_usd", event.sourceAmountUsd)
            put("channel_id", event.channelId)
            put("category", event.category)
            put("group_key", event.groupKey)
            put("has_content_intent", if (event.hasContentIntent) 1 else 0)
            put("notification_action_count", event.notificationActionCount)
        }
        return writableDatabase.insertOrThrow(TABLE, null, values)
    }

    fun stats(): RecorderStats {
        val db = readableDatabase
        var total = 0L
        var buys = 0L
        var sells = 0L
        db.rawQuery(
            """
            SELECT
                COUNT(*),
                COALESCE(SUM(CASE WHEN action='bought' THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN action='sold' THEN 1 ELSE 0 END), 0)
            FROM fomo_trade_events
            """.trimIndent(),
            null
        ).use { cursor ->
            if (cursor.moveToFirst()) {
                total = cursor.getLong(0)
                buys = cursor.getLong(1)
                sells = cursor.getLong(2)
            }
        }

        var latest: String? = null
        db.rawQuery(
            """
            SELECT action, trader, token, title, post_time_ms
            FROM fomo_trade_events
            ORDER BY event_id DESC
            LIMIT 1
            """.trimIndent(),
            null
        ).use { cursor ->
            if (cursor.moveToFirst()) {
                val action = cursor.getString(0)
                val trader = cursor.getString(1)
                val token = cursor.getString(2)
                val title = cursor.getString(3)
                val postTime = cursor.getLong(4)
                latest = listOfNotNull(
                    action,
                    trader?.let { "@$it" },
                    token ?: title,
                    "postTime=$postTime"
                ).joinToString(" | ")
            }
        }

        return RecorderStats(total, buys, sells, latest)
    }

    fun exportCsv(destination: File): Int {
        destination.parentFile?.mkdirs()
        var rows = 0
        destination.bufferedWriter().use { out ->
            out.appendLine(
                listOf(
                    "event_id",
                    "package_name",
                    "notification_key",
                    "notification_id",
                    "notification_tag",
                    "post_time_ms",
                    "captured_time_ms",
                    "title",
                    "text",
                    "big_text",
                    "selected_text",
                    "action",
                    "trader",
                    "token",
                    "market_cap_usd",
                    "source_amount_usd",
                    "channel_id",
                    "category",
                    "group_key",
                    "has_content_intent",
                    "notification_action_count",
                ).joinToString(",")
            )

            readableDatabase.rawQuery(
                "SELECT * FROM fomo_trade_events ORDER BY event_id",
                null
            ).use { cursor ->
                while (cursor.moveToNext()) {
                    val values = (0 until cursor.columnCount).map { index ->
                        if (cursor.isNull(index)) "" else cursor.getString(index)
                    }
                    out.appendLine(values.joinToString(",") { csv(it) })
                    rows += 1
                }
            }
        }
        return rows
    }

    private fun csv(value: String): String = "\"" + value.replace("\"", "\"\"") + "\""

    companion object {
        const val DB_NAME = "fomo_buy_sell.db"
        private const val DB_VERSION = 1
        private const val TABLE = "fomo_trade_events"
    }
}
