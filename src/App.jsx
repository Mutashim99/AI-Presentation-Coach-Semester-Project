import React, { useState, useRef, useEffect, useCallback } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";
import { jsPDF } from "jspdf";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Mic,
  Cpu,
  AlertTriangle,
  ScanFace,
  Smile,
  Move,
  Terminal,
  Target,
  Download,
  Eye,
} from "lucide-react";

// --- CONFIGURATION ---
const GEMINI_API_KEY = "";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

const App = () => {
  // --- STATE ---
  const [phase, setPhase] = useState("boot");
  const [isListening, setIsListening] = useState(false);
  const [countdown, setCountdown] = useState(5);
  const [reportStatus, setReportStatus] = useState(null);

  // UI Metrics (Throttled)
  const [uiMetrics, setUiMetrics] = useState({
    wpm: 0,
    volume: 0,
    readingScore: 0,
    nervousness: 0,
    smileScore: 0,
    handActivity: 0,
    words: 0,
    fillers: 0,
    gazeStability: 100,
  });

  const [uiDebug, setUiDebug] = useState({
    posture: 0,
    yaw: 0,
    pitch: 0,
    hand: 0,
  });

  const [transcript, setTranscript] = useState(["", ""]);
  const [alerts, setAlerts] = useState([]);

  // --- REFS ---
  const metricsRef = useRef({
    wpm: 0,
    volume: 0,
    readingScore: 0,
    nervousness: 0,
    smileScore: 0,
    handActivity: 0,
    words: 0,
    fillers: 0,
    gazeStability: 100,
    fillerBreakdown: {},
  });
  // NEW: Add this ref to store debug data without re-rendering
  const debugDataRef = useRef({ posture: 0, yaw: 0, pitch: 0, hand: 0 });

  const fullTranscriptRef = useRef("");

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCanvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const recognitionRef = useRef(null);

  const baselineRef = useRef({ shoulderHeight: 0, noseY: 0, noseX: 0 });
  const startTimeRef = useRef(null);
  const shouldListenRef = useRef(false);
  const isMounted = useRef(true);

  // Buffers
  const handHistoryRef = useRef([]);
  const headHistoryRef = useRef([]);
  const alertCooldownRef = useRef({});

  const FILLERS = [
    "um",
    "uh",
    "like",
    "actually",
    "basically",
    "literally",
    "so",
    "mean",
    "you know",
    "ahh",
    "umm",
    "uhh",
    "i mean",
    "yeah"
  ];

  // --- 1. BOOT ---
  useEffect(() => {
    isMounted.current = true;
    if (phase === "boot") {
      const timer = setTimeout(() => {
        if (isMounted.current) setPhase("auth");
      }, 2000);
      return () => clearTimeout(timer);
    }
    return () => {
      isMounted.current = false;
    };
  }, [phase]);

  // --- 2. UI SYNC (Throttled 5FPS) ---
  // useEffect(() => {
  //   if (phase === "live") {
  //     const interval = setInterval(() => {
  //       if (isMounted.current) setUiMetrics({ ...metricsRef.current });
  //     }, 200);
  //     return () => clearInterval(interval);
  //   }
  // }, [phase]);
  // --- 2. UI SYNC (Throttled 5FPS) ---
  useEffect(() => {
    if (phase === "live") {
      const interval = setInterval(() => {
        if (isMounted.current) {
          // Update both Metrics AND Debug data here (only 5 times a second)
          setUiMetrics({ ...metricsRef.current });
          setUiDebug({ ...debugDataRef.current });
        }
      }, 200);
      return () => clearInterval(interval);
    }
  }, [phase]);

  // --- 3. COUNTDOWN ---
  useEffect(() => {
    if (phase === "calibrating") {
      if (countdown > 0) {
        const timer = setTimeout(() => setCountdown((c) => c - 1), 1000);
        return () => clearTimeout(timer);
      } else {
        startTimeRef.current = Date.now();
        startAudioEngine();
        setPhase("live");
      }
    }
  }, [phase, countdown]);

  // --- 4. VISION ENGINE (OPTIMIZED) ---
  useEffect(() => {
    let camera = null;
    let holistic = null;

    if (["preview", "calibrating", "live"].includes(phase)) {
      holistic = new Holistic({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
      });

      // OPTIMIZATION SETTINGS:
      holistic.setOptions({
        modelComplexity: 0, // 0=Lite (Fastest), 1=Full, 2=Heavy
        smoothLandmarks: true, // Use internal C++ smoothing (faster than JS)
        enableSegmentation: false, // Don't need background segmentation
        smoothSegmentation: false,
        refineFaceLandmarks: false, // Turn off heavy iris/lip refinement
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      holistic.onResults(onResults);

      if (videoRef.current) {
        let lastFrameTime = 0;
        const TARGET_FPS = 15; // Run AI only 15 times per second
        const FRAME_DELAY = 1000 / TARGET_FPS;

        camera = new Camera(videoRef.current, {
          onFrame: async () => {
            const now = Date.now();
            // IF not enough time has passed, SKIP this frame
            if (now - lastFrameTime < FRAME_DELAY) return;

            lastFrameTime = now;

            if (videoRef.current && holistic) {
              try {
                await holistic.send({ image: videoRef.current });
              } catch (e) {}
            }
          },
          width: 640, // Keep this low resolution!
          height: 480,
        });
        camera.start();
      }
    }

    return () => {
      if (camera) camera.stop();
      if (holistic) holistic.close();
    };
  }, [phase]);

  // --- 5. RESULTS (REMOVED FILTERS & MANUAL SMOOTHING) ---
  const onResults = useCallback(
    (results) => {
      if (!canvasRef.current || !videoRef.current) return;
      const ctx = canvasRef.current.getContext("2d", { alpha: false }); // Optimize for no transparency
      const w = canvasRef.current.width;
      const h = canvasRef.current.height;

      ctx.save();

      // Mirror Transform
      ctx.scale(-1, 1);
      ctx.translate(-w, 0);

      // Draw Video - Clean, No Filters
      ctx.drawImage(results.image, 0, 0, w, h);

      // Logic
      const posed = results.poseLandmarks;
      if (posed) {
        const skeletonColor = phase === "calibrating" ? "#FFFF00" : "#00f3ff";
        drawTechSkeleton(ctx, posed, skeletonColor);

        if (phase === "calibrating") {
          captureBaseline(posed, results.faceLandmarks);
        } else if (phase === "live") {
          analyzeBody(posed);
        }
      }

      if (results.faceLandmarks && phase === "live") {
        analyzeFace(results.faceLandmarks, w, h, ctx);
      }

      ctx.restore();
    },
    [phase]
  );

  // --- 6. CALIBRATION ---
  const captureBaseline = (pose, face) => {
    const currentShoulders = (pose[11].y + pose[12].y) / 2;
    let currentNoseY = 0;
    let currentNoseX = 0;

    if (face) {
      const nose = face[1];
      const leftEar = face[234];
      const rightEar = face[454];
      const faceWidth = Math.abs(leftEar.x - rightEar.x);
      const earCenter = (leftEar.x + rightEar.x) / 2;
      currentNoseX = (nose.x - earCenter) / faceWidth;
      const earHeight = (leftEar.y + rightEar.y) / 2;
      currentNoseY = (nose.y - earHeight) / faceWidth;
    }

    if (baselineRef.current.shoulderHeight === 0) {
      baselineRef.current = {
        shoulderHeight: currentShoulders,
        noseY: currentNoseY,
        noseX: currentNoseX,
      };
    } else {
      baselineRef.current.shoulderHeight =
        (baselineRef.current.shoulderHeight + currentShoulders) / 2;
      if (face) {
        baselineRef.current.noseY =
          (baselineRef.current.noseY + currentNoseY) / 2;
        baselineRef.current.noseX =
          (baselineRef.current.noseX + currentNoseX) / 2;
      }
    }
  };

  // --- 7. BODY ANALYTICS ---
  const analyzeBody = (landmarks) => {
    const avgShoulder = (landmarks[11].y + landmarks[12].y) / 2;
    const delta = avgShoulder - baselineRef.current.shoulderHeight;

    const lw = landmarks[15];
    const rw = landmarks[16];

    // Hand Speed Logic
    const prevX = handHistoryRef.current[0]?.x || lw.x;
    const rawDiff = Math.abs(lw.x - prevX);
    const filteredDiff = rawDiff < 0.005 ? 0 : rawDiff;
    const handSpeed = filteredDiff * 100;

    handHistoryRef.current.unshift({ x: lw.x, y: lw.y });
    if (handHistoryRef.current.length > 15) handHistoryRef.current.pop();
    const activity = Math.min(handSpeed * 25, 100);

    // setUiDebug((p) => ({
    //   ...p,
    //   posture: delta.toFixed(3),
    //   hand: activity.toFixed(0),
    // }));
    // Write to Ref instead of State to prevent lag
    debugDataRef.current.posture = delta.toFixed(3);
    debugDataRef.current.hand = activity.toFixed(0);
    metricsRef.current.handActivity = Math.round(activity);

    if (delta > 0.04) triggerAlert("POSTURE: SIT UP!", 2000);
    if (activity < 2 && metricsRef.current.words > 10)
      triggerAlert("HANDS: TOO STIFF", 4000);
    if (activity > 85) triggerAlert("HANDS: DISTRACTING", 4000);
  };

  // --- 8. FACE ANALYTICS ---
  const analyzeFace = (landmarks, w, h, ctx) => {
    const nose = landmarks[1];
    const leftEar = landmarks[234];
    const rightEar = landmarks[454];
    const faceWidth = Math.abs(leftEar.x - rightEar.x);

    const earCenter = (leftEar.x + rightEar.x) / 2;
    const currentNoseX = (nose.x - earCenter) / faceWidth;
    const earHeight = (leftEar.y + rightEar.y) / 2;
    const currentNoseY = (nose.y - earHeight) / faceWidth;

    const yawDrift = Math.abs(currentNoseX - baselineRef.current.noseX);
    const pitchDrift = currentNoseY - baselineRef.current.noseY;

    headHistoryRef.current.push(currentNoseX);
    if (headHistoryRef.current.length > 20) headHistoryRef.current.shift();
    let variance = 0;
    if (headHistoryRef.current.length > 5) {
      const mean =
        headHistoryRef.current.reduce((a, b) => a + b, 0) /
        headHistoryRef.current.length;
      variance = headHistoryRef.current.reduce(
        (a, b) => a + Math.pow(b - mean, 2),
        0
      );
    }
    const stability = Math.max(0, 100 - variance * 20000);
    metricsRef.current.gazeStability = Math.round(stability);

    // setUiDebug((p) => ({
    //   ...p,
    //   yaw: yawDrift.toFixed(2),
    //   pitch: pitchDrift.toFixed(2),
    // }));
    // Write to Ref instead of State
    debugDataRef.current.yaw = yawDrift.toFixed(2);
    debugDataRef.current.pitch = pitchDrift.toFixed(2);
    metricsRef.current.readingScore = Math.max(
      0,
      100 - Math.round(pitchDrift * 300)
    );

    if (yawDrift > 0.15) triggerAlert("FACE: TURN TO CAMERA", 1500);
    if (pitchDrift > 0.12) triggerAlert("FACE: READING NOTES?", 1500);
    if (stability < 40 && metricsRef.current.words > 10)
      triggerAlert("FOCUS: HEAD MOVING", 2000);

    // Smile
    const mouthW = Math.abs(landmarks[61].x - landmarks[291].x);
    const mouthH = Math.abs(landmarks[13].y - landmarks[14].y);
    const smileRatio = mouthW / mouthH;
    metricsRef.current.smileScore = smileRatio > 1.8 ? 100 : 0;

    if (smileRatio < 1.5 && metricsRef.current.words > 15) {
      if (Math.random() > 0.99) triggerAlert("😊 REMEMBER TO SMILE", 4000);
    }

    const color = yawDrift > 0.15 || pitchDrift > 0.12 ? "#ff2a2a" : "#00f3ff";
    drawSciFiHUD(ctx, landmarks, w, h, color);
  };

  const triggerAlert = (msg, cooldown = 3000) => {
    const now = Date.now();
    const lastTrigger = alertCooldownRef.current[msg] || 0;
    if (now - lastTrigger > cooldown) {
      alertCooldownRef.current[msg] = now;
      setAlerts((prev) => {
        const newAlerts = [...prev, msg].slice(-3);
        setTimeout(
          () => setAlerts((curr) => curr.filter((a) => a !== msg)),
          2000
        );
        return newAlerts;
      });
    }
  };

  // --- 9. AUDIO ENGINE ---
  const startAudioEngine = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioCtx.createAnalyser();
      const source = audioCtx.createMediaStreamSource(stream);
      analyser.fftSize = 256;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;
      drawWaveform();

      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        if (recognitionRef.current) recognitionRef.current.stop();
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";
        shouldListenRef.current = true;

        recognition.onstart = () => setIsListening(true);
        recognition.onend = () => {
          setIsListening(false);
          if (shouldListenRef.current) {
            setTimeout(() => {
              try {
                recognition.start();
              } catch (e) {}
            }, 100);
          }
        };
        recognition.onresult = handleSpeechResult;
        recognition.start();
        recognitionRef.current = recognition;
      }
    } catch (e) {
      console.error("Audio Fail", e);
    }
  };

  const stopAudioEngine = () => {
    shouldListenRef.current = false;
    if (recognitionRef.current) recognitionRef.current.stop();
  };

  const drawWaveform = () => {
    if (!audioCanvasRef.current || !analyserRef.current) return;
    const canvas = audioCanvasRef.current;
    const ctx = canvas.getContext("2d");
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      if (!analyserRef.current || !isMounted.current) return;
      analyserRef.current.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;

      if (phase === "live" && avg > 0) {
        if (avg < 15 && avg > 1) triggerAlert("🔊 SPEAK LOUDER", 3000);
        if (avg > 80) triggerAlert("📢 TOO LOUD", 3000);
      }
      const barWidth = (canvas.width / dataArray.length) * 2.5;
      let x = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const barH = dataArray[i] / 2;
        ctx.fillStyle = `rgba(0, 243, 255, ${barH / 100})`;
        ctx.fillRect(x, canvas.height - barH, barWidth, barH);
        x += barWidth + 1;
      }
      metricsRef.current.volume = Math.round(avg);
      requestAnimationFrame(render);
    };
    render();
  };

  const handleSpeechResult = (event) => {
    let interimContent = "";
    let newFinalContent = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        newFinalContent += event.results[i][0].transcript + " ";
      } else {
        interimContent += event.results[i][0].transcript;
      }
    }

    if (newFinalContent) fullTranscriptRef.current += newFinalContent;
    setTranscript([fullTranscriptRef.current, interimContent]);

    const totalText = (fullTranscriptRef.current + interimContent).trim();
    const wordsArray = totalText.split(/\s+/).filter((w) => w.length > 0);
    const wordCount = wordsArray.length;
    const mins = (Date.now() - startTimeRef.current) / 60000;

    let wpm = 0;
    if (mins > 0.08 && wordCount > 0) wpm = Math.round(wordCount / mins);

    let fillerCount = 0;
    const currentFillerStats = {};

    FILLERS.forEach((filler) => {
      const regex = new RegExp(`\\b${filler}\\b`, "gi");
      const matches = totalText.match(regex);
      if (matches) {
        fillerCount += matches.length;
        currentFillerStats[filler] = matches.length;
      }
    });

    metricsRef.current.wpm = wpm;
    metricsRef.current.words = wordCount;
    metricsRef.current.fillers = fillerCount;
    metricsRef.current.fillerBreakdown = currentFillerStats;

    if (interimContent) {
      const lastWord = interimContent.trim().split(" ").pop().toLowerCase();
      if (
        FILLERS.includes(lastWord) &&
        !alertCooldownRef.current[`filler-${lastWord}`]
      ) {
        triggerAlert(`FILLER: ${lastWord.toUpperCase()}`, 1500);
        alertCooldownRef.current[`filler-${lastWord}`] = Date.now();
      }
    }
  };

  // --- 10. VISUALS (CLEAN) ---
  const drawTechSkeleton = (ctx, landmarks, color) => {
    const connections = [
      [11, 12],
      [12, 24],
      [24, 23],
      [23, 11],
    ];
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      ctx.beginPath();
      ctx.moveTo(p1.x * ctx.canvas.width, p1.y * ctx.canvas.height);
      ctx.lineTo(p2.x * ctx.canvas.width, p2.y * ctx.canvas.height);
      ctx.stroke();
    });
  };

  const drawSciFiHUD = (ctx, landmarks, w, h, color) => {
    const x1 = landmarks[234].x * w;
    const x2 = landmarks[454].x * w;
    const y1 = landmarks[10].y * h - 40;
    const y2 = landmarks[152].y * h + 40;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
    ctx.shadowColor = color;
    const L = 20;

    ctx.beginPath();
    ctx.moveTo(x1, y1 + L);
    ctx.lineTo(x1, y1);
    ctx.lineTo(x1 + L, y1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2 - L, y1);
    ctx.lineTo(x2, y1);
    ctx.lineTo(x2, y1 + L);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y2 - L);
    ctx.lineTo(x1, y2);
    ctx.lineTo(x1 + L, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2 - L, y2);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x2, y2 - L);
    ctx.stroke();
    ctx.shadowBlur = 0;
  };

  // --- 11. REPORT GENERATION ---
  const generateLocalReport = (m, duration) => {
    let summary = `Session Duration: ${duration} min. Pacing: ${m.wpm} WPM. `;
    let grade = "B";
    let title = "The Developing Speaker";
    let tips = ["Practice varying your tone."];

    if (m.wpm > 130) {
      grade = "A-";
      title = "The Energetic Presenter";
      summary += "Your energy was high. ";
      tips.push("Pause for emphasis.");
    } else if (m.wpm < 100) {
      grade = "C+";
      title = "The Thoughtful Observer";
      summary += "Delivery was slow. ";
      tips.push("Increase urgency.");
    }

    if (m.fillers > 3) {
      summary += `Detected ${m.fillers} fillers. `;
      tips.push("Pause instead of using fillers.");
    }
    return { grade, title, summary, tips };
  };

  const generateReport = async () => {
    setPhase("processing");
    setReportStatus("analyzing");
    stopAudioEngine();

    const duration = ((Date.now() - startTimeRef.current) / 60000).toFixed(2);
    const m = metricsRef.current;
    const safeTranscript =
      fullTranscriptRef.current || "No speech recorded during this session.";

    // AI ANALYSIS
    let ai = {
      grade: "B",
      title: "Developing Speaker",
      summary: "Analysis complete.",
      tips: ["Practice makes perfect."],
    };
    try {
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
      const prompt = `Executive Coach Analysis. Stats: ${duration}m, ${
        m.wpm
      } WPM, ${m.fillers} fillers. Transcript: "${safeTranscript.substring(
        0,
        800
      )}". JSON { "grade": "S/A/B", "title": "Archetype", "summary": "Short summary", "tips": ["Tip 1", "Tip 2"] }`;
      const res = await model.generateContent(prompt);
      const text = res.response
        .text()
        .replace(/```json/g, "")
        .replace(/```/g, "");
      ai = JSON.parse(text);
      setReportStatus("done");
    } catch (e) {
      console.log(e);
    }

    // --- PDF GENERATION START ---
    const doc = new jsPDF();
    const bg = [15, 23, 42]; // Slate 950
    const accent = [6, 182, 212]; // Cyan 500
    const white = [255, 255, 255];
    const grey = [148, 163, 184];

    // --- PAGE 1: ANALYTICS ---
    doc.setFillColor(...bg);
    doc.rect(0, 0, 210, 297, "F");

    // Header
    doc.setFillColor(30, 41, 59);
    doc.rect(0, 0, 210, 40, "F");
    doc.setTextColor(...accent);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.text("AI PRESENTATION COACH", 20, 20);
    doc.setFontSize(10);
    doc.setTextColor(...white);
    doc.text("COMMUNICATION ANALYSIS", 20, 30);
    doc.text(`DATE: ${new Date().toLocaleDateString()}`, 150, 20);
    doc.text(`DURATION: ${duration} MIN`, 150, 30);

    // Big Grade Section
    doc.setDrawColor(...accent);
    doc.setLineWidth(1);
    doc.roundedRect(20, 50, 170, 55, 3, 3);

    doc.setFontSize(60);
    doc.setTextColor(...accent);
    doc.text(ai.grade, 30, 90);

    doc.setFontSize(16);
    doc.setTextColor(...white);
    doc.text(ai.title.toUpperCase(), 70, 70);

    doc.setFontSize(10);
    doc.setTextColor(...grey);
    // CRITICAL FIX: Split text to fit
    const summaryLines = doc.splitTextToSize(ai.summary, 110);
    doc.text(summaryLines, 70, 80);

    // Visual Metrics Bars
    let y = 120;
    const drawBar = (label, value, max, color) => {
      doc.setFontSize(10);
      doc.setTextColor(...white);
      doc.text(label, 20, y);

      // Background Bar
      doc.setFillColor(51, 65, 85);
      doc.roundedRect(60, y - 4, 100, 4, 1, 1, "F");

      // Fill Bar
      doc.setFillColor(...color);
      const width = Math.min((value / max) * 100, 100);
      if (width > 0) doc.roundedRect(60, y - 4, width, 4, 1, 1, "F");

      doc.text(`${value}`, 170, y);
      y += 12;
    };

    drawBar("Pacing (WPM)", m.wpm, 180, accent);
    drawBar("Volume Level", m.volume, 100, [168, 85, 247]);
    drawBar("Filler Words", m.fillers, 20, [239, 68, 68]);
    drawBar("Head Stability", m.gazeStability, 100, [34, 197, 94]);
    drawBar("Smile Score", m.smileScore, 100, [249, 115, 22]);

    // Tips Section
    y += 10;
    doc.setDrawColor(51, 65, 85);
    doc.line(20, y, 190, y);
    y += 15;

    doc.setTextColor(...accent);
    doc.setFontSize(14);
    doc.text("COACHING TIPS", 20, y);
    y += 10;

    doc.setTextColor(...white);
    doc.setFontSize(10);
    if (ai.tips) {
      ai.tips.forEach((t) => {
        // CRITICAL FIX: Split tips text
        const lines = doc.splitTextToSize(`• ${t}`, 170);
        doc.text(lines, 20, y);
        y += lines.length * 5 + 3;
      });
    }

    // --- PAGE 2: TRANSCRIPT ---
    doc.addPage();
    doc.setFillColor(...bg);
    doc.rect(0, 0, 210, 297, "F");

    // Transcript Header
    doc.setFillColor(30, 41, 59);
    doc.rect(0, 0, 210, 30, "F");
    doc.setTextColor(...accent);
    doc.setFontSize(16);
    doc.text("SESSION TRANSCRIPT", 20, 20);

    // Transcript Body
    doc.setTextColor(200, 200, 200);
    doc.setFont("courier", "normal");
    doc.setFontSize(10);

    const splitTranscript = doc.splitTextToSize(safeTranscript, 170);
    let lineY = 50;

    // Loop through lines to handle page breaks for very long transcripts
    for (let i = 0; i < splitTranscript.length; i++) {
      if (lineY > 280) {
        doc.addPage();
        doc.setFillColor(...bg);
        doc.rect(0, 0, 210, 297, "F");
        doc.setTextColor(200, 200, 200);
        lineY = 20;
      }
      doc.text(splitTranscript[i], 20, lineY);
      lineY += 5;
    }

    doc.save("SmartStance_Report_Pro.pdf");
    setPhase("boot");
  };

  // --- 12. UI ---
  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans overflow-hidden selection:bg-cyan-500/30">
      {/* GLOBAL BACKGROUND ANIMATION */}
      {/* <div
        className="absolute inset-0 opacity-20 pointer-events-none z-0"
        style={{
          backgroundImage:
            "linear-gradient(rgba(6,182,212,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.1) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      /> */}

      {/* PHASE 1: BOOT */}
      <AnimatePresence>
        {phase === "boot" && (
          <motion.div
            exit={{ opacity: 0 }}
            className="flex h-screen items-center justify-center bg-black z-50"
          >
            <div className="text-center relative">
              <div className="absolute inset-0 bg-cyan-500/20 blur-xl rounded-full animate-pulse"></div>
              <Cpu className="w-24 h-24 text-cyan-400 relative z-10 animate-spin-slow mb-6 mx-auto" />
              <div className="h-1 w-64 bg-slate-800 rounded-full overflow-hidden mx-auto">
                <motion.div
                  className="h-full bg-cyan-500 box-shadow-[0_0_10px_#06b6d4]"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 2 }}
                />
              </div>
              <p className="mt-4 font-mono text-cyan-500 text-sm tracking-[0.3em]">
                INITIALIZING NEURAL LINK...
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* PHASE 2: AUTH */}
      {phase === "auth" && (
        <div className="flex h-screen items-center justify-center relative z-40">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-slate-900/80 backdrop-blur-xl border border-slate-700 p-10 rounded-3xl w-full max-w-lg shadow-2xl relative overflow-hidden"
          >
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent"></div>

            <div className="flex items-center gap-4 mb-8">
              <div className="p-3 bg-cyan-500/10 rounded-xl border border-cyan-500/30">
                <ScanFace size={32} className="text-cyan-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white tracking-tight">
                  AI PRESENTATION COACH
                </h1>
                <p className="text-cyan-500 text-sm font-mono tracking-widest">
                  PRO EDITION v2.0
                </p>
              </div>
            </div>

            <p className="text-slate-400 mb-8 leading-relaxed">
              Advanced AI communication analysis system ready. Ensure camera and
              microphone access for real-time telemetry.
            </p>

            <button
              onClick={() => setPhase("preview")}
              className="group w-full py-4 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-xl font-bold text-lg tracking-wide shadow-lg hover:shadow-cyan-500/25 transition-all flex items-center justify-center gap-2"
            >
              <Target className="group-hover:rotate-180 transition-transform" />{" "}
              ENGAGE SYSTEM
            </button>
          </motion.div>
        </div>
      )}

      {/* PHASE 3: MAIN INTERFACE */}
      {["preview", "calibrating", "live"].includes(phase) && (
        <div className="relative h-screen flex">
          {/* VIDEO LAYER */}
          <div className="absolute inset-0 bg-black">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover opacity-60"
              playsInline
              muted
              autoPlay
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover"
              width={640}
              height={480}
            />

            {/* GRID OVERLAY */}
            <div
              className="absolute inset-0 pointer-events-none opacity-20"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)",
                backgroundSize: "100px 100px",
              }}
            />
          </div>

          {/* LEFT SIDEBAR - TELEMETRY */}
          <div
            className={`w-80 h-full z-20 p-6 flex flex-col gap-4 transition-transform duration-500 ${
              phase === "live" ? "translate-x-0" : "-translate-x-full"
            }`}
          >
            <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-md border border-slate-700/50 shadow-xl">
              <div className="flex items-center gap-2 text-cyan-400 mb-2 border-b border-white/10 pb-2">
                <Activity size={16} />{" "}
                <span className="text-xs font-bold tracking-widest">
                  PACING
                </span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-mono font-bold text-white">
                  {uiMetrics.wpm}
                </span>
                <span className="text-xs text-slate-400">WPM</span>
              </div>
              <div className="w-full bg-slate-800 h-1.5 mt-2 rounded-full overflow-hidden">
                <motion.div
                  className="h-full bg-cyan-400"
                  animate={{ width: `${Math.min(uiMetrics.wpm / 2, 100)}%` }}
                />
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-md border border-slate-700/50 shadow-xl">
              <div className="flex items-center gap-2 text-purple-400 mb-2 border-b border-white/10 pb-2">
                <AlertTriangle size={16} />{" "}
                <span className="text-xs font-bold tracking-widest">
                  FILLERS
                </span>
              </div>
              <span className="text-4xl font-mono font-bold text-white">
                {uiMetrics.fillers}
              </span>
            </div>

            <div className="mt-auto p-4 rounded-xl bg-slate-900/80 backdrop-blur-md border border-slate-700/50">
              <div className="flex items-center gap-2 text-green-400 mb-3">
                <Mic size={16} />{" "}
                <span className="text-xs font-bold tracking-widest">
                  AUDIO INPUT
                </span>
              </div>
              <canvas
                ref={audioCanvasRef}
                width={250}
                height={60}
                className="w-full"
              />
            </div>
          </div>

          {/* CENTER - ALERTS */}
          <div className="absolute top-10 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center w-full max-w-lg pointer-events-none">
            <AnimatePresence>
              {alerts.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ y: -50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="mb-2 px-6 py-3 bg-red-500/90 backdrop-blur-md text-white font-bold rounded-lg shadow-lg border border-red-400 flex items-center gap-3"
                >
                  <AlertTriangle className="animate-pulse" /> {msg}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

          {/* RIGHT SIDEBAR - METRICS */}
          <div
            className={`absolute right-0 top-0 h-full w-72 p-6 flex flex-col gap-4 transition-transform duration-500 ${
              phase === "live" ? "translate-x-0" : "translate-x-full"
            }`}
          >
            <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-md border border-slate-700/50 shadow-xl space-y-4">
              <h3 className="text-xs font-bold text-slate-400 tracking-widest border-b border-white/10 pb-2">
                BIO-METRICS
              </h3>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-300 flex items-center gap-2">
                  <Eye size={14} /> FOCUS
                </span>
                <span
                  className={`text-sm font-bold ${
                    uiMetrics.gazeStability > 50
                      ? "text-green-400"
                      : "text-red-400"
                  }`}
                >
                  {uiMetrics.gazeStability}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-300 flex items-center gap-2">
                  <Smile size={14} /> WARMTH
                </span>
                <span
                  className={`text-sm font-bold ${
                    uiMetrics.smileScore > 50
                      ? "text-green-400"
                      : "text-slate-400"
                  }`}
                >
                  {uiMetrics.smileScore}%
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-300 flex items-center gap-2">
                  <Move size={14} /> GESTURE
                </span>
                <span className="text-sm font-bold text-cyan-400">
                  {uiMetrics.handActivity}%
                </span>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 backdrop-blur-md border border-slate-700/50 shadow-xl mt-auto mb-20">
              <div className="flex items-center gap-2 text-cyan-400 mb-2">
                <Terminal size={14} />{" "}
                <span className="text-[10px] font-bold tracking-widest">
                  DEBUG STREAM
                </span>
              </div>
              <div className="font-mono text-[10px] text-slate-400 space-y-1">
                <p>POSTURE_DELTA: {uiDebug.posture}</p>
                <p>HEAD_YAW: {uiDebug.yaw}</p>
                <p>VOICE_ENG: {isListening ? "ACTIVE" : "IDLE"}</p>
                <p>SYS_STATUS: OPTIMAL</p>
              </div>
            </div>
          </div>

          {/* BOTTOM BAR - CONTROLS */}
          <div className="absolute bottom-0 left-0 w-full p-8 flex justify-center items-end bg-gradient-to-t from-black/90 to-transparent z-30">
            {phase === "live" ? (
              <div className="w-full max-w-4xl flex gap-6 items-end">
                <div className="flex-1 bg-slate-900/50 backdrop-blur border border-white/10 rounded-lg p-3 h-20 overflow-hidden relative">
                  <p className="text-cyan-200/80 text-sm font-medium leading-relaxed">
                    {transcript[0]}{" "}
                    <span className="text-white bg-cyan-500/20">
                      {transcript[1]}
                    </span>
                  </p>
                </div>
                <button
                  onClick={generateReport}
                  className="px-8 py-4 bg-red-600 hover:bg-red-500 rounded-lg font-bold text-white shadow-lg shadow-red-900/50 transition-all flex items-center gap-2 shrink-0"
                >
                  <Download size={20} /> END SESSION
                </button>
              </div>
            ) : phase === "preview" ? (
              <button
                onClick={() => setPhase("calibrating")}
                className="px-12 py-5 bg-cyan-500 text-slate-900 font-bold text-xl rounded-full shadow-[0_0_30px_rgba(6,182,212,0.6)] animate-pulse"
              >
                START CALIBRATION
              </button>
            ) : null}
          </div>

          {/* CALIBRATION OVERLAY */}
          {phase === "calibrating" && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-sm z-50">
              <motion.div
                key={countdown}
                initial={{ scale: 2, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="text-center"
              >
                <div className="text-[12rem] font-bold text-transparent bg-clip-text bg-gradient-to-b from-cyan-300 to-cyan-600 leading-none">
                  {countdown}
                </div>
                <p className="text-white text-xl font-bold tracking-[0.5em] mt-4">
                  CALIBRATING SENSORS
                </p>
                <p className="text-cyan-400 mt-2">ASSUME NEUTRAL POSTURE</p>
              </motion.div>
            </div>
          )}
        </div>
      )}

      {/* PHASE 4: REPORT GENERATION */}
      {phase === "processing" && (
        <div className="flex h-screen items-center justify-center bg-slate-900 z-50">
          <div className="text-center">
            <div className="w-20 h-20 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mx-auto mb-6"></div>
            <h2 className="text-2xl font-bold text-white mb-2">
              COMPILING REPORT
            </h2>
            <p className="text-slate-400 animate-pulse">
              {reportStatus === "analyzing"
                ? "CONSULTING AI NEURAL NET..."
                : "GENERATING SECURE PDF..."}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
