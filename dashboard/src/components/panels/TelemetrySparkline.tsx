'use client';

// real-time telemetry sparkline rendering PPS and bandwidth metrics

import React, { useEffect, useRef } from 'react';

interface TelemetrySparklineProps {
  label: string;
  value: string | number;
  unit: string;
  color?: string;
  trend?: number[];
  height?: number;
}

export function TelemetrySparkline({
  label,
  value,
  unit,
  color = '#00f0ff',
  trend = [24, 30, 45, 38, 52, 60, 58, 72, 68, 85, 90, 82, 94],
  height = 28,
}: TelemetrySparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (trend.length < 2) return;

    const max = Math.max(...trend, 100);
    const min = Math.min(...trend, 0);
    const range = max - min || 1;

    // background gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, `${color}40`);
    grad.addColorStop(1, `${color}00`);

    ctx.beginPath();
    trend.forEach((val, i) => {
      const x = (i / (trend.length - 1)) * w;
      const y = h - ((val - min) / range) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    // close path for gradient fill
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // stroke line
    ctx.beginPath();
    trend.forEach((val, i) => {
      const x = (i / (trend.length - 1)) * w;
      const y = h - ((val - min) / range) * (h - 4) - 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // highlight latest point
    const lastX = w;
    const lastY = h - ((trend[trend.length - 1] - min) / range) * (h - 4) - 2;
    ctx.beginPath();
    ctx.arc(lastX - 2, lastY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.stroke();
  }, [trend, color]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: 'var(--font-mono)' }}>
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <span style={{ fontSize: 9, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </span>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)' }}>{value}</span>
          <span style={{ fontSize: 9, color: 'var(--text-2)' }}>{unit}</span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={72}
        height={height}
        style={{ display: 'block', width: 72, height }}
      />
    </div>
  );
}
