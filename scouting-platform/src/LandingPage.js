import React, { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import footballIcon from "./football.png";

function LandingPage() {
  const navigate = useNavigate();
  const shapesRef = useRef([]);

  // Animate floating footballs
  useEffect(() => {
    const animateShapes = () => {
      shapesRef.current.forEach((shape, idx) => {
        if (!shape) return;
        let top = parseFloat(shape.style.top);
        top += 0.15 + idx * 0.03;
        if (top > window.innerHeight) top = -50;
        shape.style.top = `${top}px`;
      });
      requestAnimationFrame(animateShapes);
    };
    animateShapes();
  }, []);

  const numShapes = 12;

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        position: "relative",
        overflow: "hidden",
        background: "linear-gradient(to bottom, #0b3d91, #1f77b4)",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        color: "#fff",
      }}
    >
      {/* Floating footballs */}
      {Array.from({ length: numShapes }).map((_, i) => (
        <img
          key={i}
          ref={(el) => (shapesRef.current[i] = el)}
          src={footballIcon}
          alt="football"
          style={{
            position: "absolute",
            top: `${Math.random() * window.innerHeight}px`,
            left: `${Math.random() * window.innerWidth}px`,
            width: `${18 + Math.random() * 15}px`,
            opacity: 0.2 + Math.random() * 0.25,
            pointerEvents: "none",
            transform: `rotate(${Math.random() * 360}deg)`,
          }}
        />
      ))}

      {/* HERO */}
      <div
        style={{
          position: "relative",
          zIndex: 2,
          textAlign: "center",
          maxWidth: 700,
          margin: "140px auto 0",
          padding: "0 15px",
        }}
      >
        <h1
          style={{
            fontSize: "3rem",
            fontWeight: "bold",
            marginBottom: 20,
            textShadow: "2px 2px 12px rgba(0,0,0,0.4)",
          }}
        >
          Professional Scouting & Analysis
        </h1>

        <p
          style={{
            fontSize: "1.2rem",
            marginBottom: 35,
            textShadow: "1px 1px 6px rgba(0,0,0,0.3)",
          }}
        >
          Upload your CSV data, explore detailed player metrics, compare
          performances, and generate professional scouting reports.
        </p>

        <button
          onClick={() => navigate("/scout-report")}
          style={{
            background: "#ffb74d",
            color: "#1a4d8f",
            border: "none",
            padding: "15px 45px",
            borderRadius: 10,
            fontSize: 18,
            fontWeight: "bold",
            cursor: "pointer",
            boxShadow: "0 6px 14px rgba(0,0,0,0.3)",
          }}
        >
          Go to Scout Reports
        </button>

        <p style={{ fontSize: "0.85rem", marginTop: 18, color: "#ddd" }}>
          Supported file type: CSV with player metrics
        </p>
      </div>
    </div>
  );
}

export default LandingPage;

