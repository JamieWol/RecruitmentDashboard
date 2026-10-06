import React from "react";
import { supabase } from "./supabaseClient";
import { clubBadgePathFor } from "./clubBadgePaths";

export const ClubBadge = ({ club, externalUrl, size = 22 }) => {
  const path = clubBadgePathFor(club);
  const storageUrl = path
    ? supabase.storage.from("Club Badges").getPublicUrl(path).data?.publicUrl
    : "";
  const src = storageUrl || externalUrl;
  if (!src || !club || String(club).trim() === "—") return null;
  return (
    <img
      src={src}
      alt=""
      aria-hidden="true"
      onError={(event) => {
        if (externalUrl && event.currentTarget.src !== externalUrl) event.currentTarget.src = externalUrl;
        else event.currentTarget.style.display = "none";
      }}
      style={{ width: size, height: size, objectFit: "contain", flex: "0 0 auto", verticalAlign: "middle" }}
    />
  );
};

export const ClubName = ({ club, fallback = "Club not added", externalUrl, size = 20 }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 6, verticalAlign: "middle" }}>
    <ClubBadge club={club} externalUrl={externalUrl} size={size} />
    <span>{club || fallback}</span>
  </span>
);
