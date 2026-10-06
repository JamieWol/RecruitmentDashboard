import { ClubName } from "./ClubBadge";
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const storageKey = "squadPlans";
const readPlans = () => {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
};
const formatDate = (value) => {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : `Updated ${date.toLocaleDateString()}`;
};

export default function SquadPlansPage() {
  const navigate = useNavigate();
  const [plans, setPlans] = useState([]);
  useEffect(() => {
    setPlans(readPlans().sort((a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0)));
  }, []);
  const createPlan = () => navigate(`/squad-plan?new=${Date.now()}`);
  const deletePlan = (event, id) => {
    event.stopPropagation();
    if (!window.confirm("Delete this squad plan?")) return;
    const next = readPlans().filter((plan) => String(plan.id) !== String(id));
    localStorage.setItem(storageKey, JSON.stringify(next));
    setPlans(next);
  };
  return <main className="sr-page sr-shortlist-view sr-squad-plans-page">
    <section className="sr-dashboard-head"><div><div className="sr-kicker">SQUAD PLANS</div><h1>Your Squad Plans</h1><p>Open a saved squad plan or create a new one.</p></div><div className="sr-shortlist-head-actions"><button type="button" className="sr-cyan" onClick={createPlan}>Create New Squad</button></div></section>
    {plans.length === 0 ? <div className="sr-empty"><p>You have not saved a squad plan yet.</p><button type="button" className="sr-cyan" onClick={createPlan}>Create your first squad</button></div> : <div className="sr-grid">{plans.map((plan) => <article className="sr-card sr-squad-plan-card" key={String(plan.id)} onClick={() => navigate(`/squad-plan?plan=${encodeURIComponent(plan.id)}`)}><div className="sr-card-top"><span className="sr-card-status">SQUAD PLAN</span><button type="button" className="sr-trash" onClick={(event) => deletePlan(event, plan.id)}>Delete</button></div><h3>{plan.name || "Unnamed squad"}</h3><p><ClubName club={plan.club} fallback="No club selected" /></p><small>{plan.formation || "No formation selected"}</small><small>{Array.isArray(plan.players) ? plan.players.length : 0} players</small>{formatDate(plan.updatedAt) && <small>{formatDate(plan.updatedAt)}</small>}</article>)}</div>}
  </main>;
}
