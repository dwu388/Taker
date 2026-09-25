package com.dwu.fomorecorder

import android.app.Notification
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import android.util.Log
import java.util.concurrent.Executors

class FomoNotificationListener : NotificationListenerService() {
    private val writer = Executors.newSingleThreadExecutor()
    private lateinit var db: FomoRecorderDb

    override fun onCreate() {
        super.onCreate()
        db = FomoRecorderDb(this)
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        if (sbn.packageName != FOMO_PACKAGE) return

        val notification = sbn.notification
        val extras = notification.extras

        val title = extras.getCharSequence(Notification.EXTRA_TITLE)
            ?.toString()
            .orEmpty()

        val text = extras.getCharSequence(Notification.EXTRA_TEXT)
            ?.toString()
            .orEmpty()

        val bigText = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)
            ?.toString()
            ?.takeIf { it.isNotBlank() }

        val textLines = extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)
            ?.joinToString("\n") { it.toString() }
            ?.takeIf { it.isNotBlank() }

        val selectedText = when {
            !bigText.isNullOrBlank() -> bigText
            text.isNotBlank() -> text
            !textLines.isNullOrBlank() -> textLines
            else -> ""
        }

        val parsed = TradeParser.parse(title, selectedText) ?: return

        val event = TradeEvent(
            packageName = sbn.packageName,
            notificationKey = sbn.key,
            notificationId = sbn.id,
            notificationTag = sbn.tag,
            postTimeMs = sbn.postTime,
            capturedTimeMs = System.currentTimeMillis(),
            title = title,
            text = text,
            bigText = bigText,
            selectedText = selectedText,
            action = parsed.action,
            trader = parsed.trader,
            token = parsed.token,
            marketCapUsd = parsed.marketCapUsd,
            sourceAmountUsd = parsed.sourceAmountUsd,
            channelId = notification.channelId,
            category = notification.category,
            groupKey = sbn.groupKey,
            hasContentIntent = notification.contentIntent != null,
            notificationActionCount = notification.actions?.size ?: 0,
        )

        writer.execute {
            runCatching { db.insert(event) }
                .onFailure { error ->
                    Log.e(TAG, "Failed to record Fomo ${event.action} notification", error)
                }
        }
    }

    override fun onDestroy() {
        writer.shutdown()
        db.close()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "FomoRecorder"
        private const val FOMO_PACKAGE = "family.fomo.app"
    }
}
