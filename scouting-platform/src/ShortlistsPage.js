import { ClubName } from "./ClubBadge";
import React, { useEffect, useMemo, useState } from "react";
import { supabase } from "./supabaseClient";
import { useAuth } from "./AuthContext";
const formations = [
    "4-1-4-1",
    "4-2-2-2",
    "4-2-3-1",
    "4-3-1-2",
    "4-3-2-1",
    "4-3-3",
    "4-4-1-1",
    "4-4-2",
    "4-4-2 (Diamond)",
    "4-5-1",
    "3-4-3",
    "3-5-2",
  ],
  formationRows = {
    "4-1-4-1": [
      ["LW", "CF", "RW"],
      ["LM", "CM", "CM", "RM"],
      ["DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-2-2-2": [
      ["CF", "CF"],
      ["AM", "AM"],
      ["DM", "DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-2-3-1": [
      ["ST"],
      ["LW", "AM", "RW"],
      ["DM", "DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-1-2": [
      ["ST", "ST"],
      ["AM"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-2-1": [
      ["ST"],
      ["AM", "AM"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-3-3": [
      ["LW", "CF", "RW"],
      ["LCM", "CM", "RCM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-1-1": [
      ["ST"],
      ["AM"],
      ["LM", "LCM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-2": [
      ["ST", "ST"],
      ["LM", "LCM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-4-2 (Diamond)": [
      ["ST", "ST"],
      ["AM"],
      ["DM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "4-5-1": [
      ["ST"],
      ["LM", "LCM", "CM", "RCM", "RM"],
      ["LB", "LCB", "RCB", "RB"],
      ["GK"],
    ],
    "3-4-3": [
      ["LW", "CF", "RW"],
      ["LM", "LCM", "RCM", "RM"],
      ["LCB", "CB", "RCB"],
      ["GK"],
    ],
    "3-5-2": [
      ["ST", "ST"],
      ["LM", "CM", "CM", "RM"],
      ["LCB", "CB", "RCB"],
      ["GK"],
    ],
  },
  flexRows = [
    ["LW", "CF", "RW"],
    ["LM", "AM", "RM"],
    ["LCM", "CM", "RCM"],
    ["LB", "LCB", "CB", "RCB", "RB"],
    ["GK"],
  ];
const shortlistPhoto = (name, lower = true) =>
  `https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/${String(
    name || "",
  )
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((x) =>
      x
        .normalize("NFD")
        .replace(/[̀-ͯ]/g, "")
        .replace(/[^a-zA-Z0-9]+/g, "_")
        [lower ? "toLowerCase" : "toString"](),
    )
    .join("_")}.png`;
const shortlistPhotoCandidates = (name) => {
  const raw = String(name || "").trim();
  const parts = raw.split(/\s+/).filter(Boolean);
  const display = parts.length > 2 ? `${parts[0]} ${parts.at(-1)}` : raw;
  const twoNameVariants = parts.length > 2 ? parts.slice(0, -1).map((_, index) => parts.slice(index, index + 2).join(" ")) : [];
  const nameVariants = [raw, ...twoNameVariants, display];
  const rawBases = nameVariants.map((value) => value.replace(/[^\p{L}\p{N}]+/gu, "_").replace(/^_+|_+$/g, ""));
  const normalizedBases = rawBases.map((value) => value.normalize("NFD").replace(/[̀-ͯ]/g, ""));
  const legacyBases = nameVariants.map((value) => value.trim().replace(/[^a-z0-9]/gi, "_").toLowerCase().replace(/^_+|_+$/g, ""));
  const bases = [...new Set([...rawBases, ...normalizedBases, ...legacyBases].filter(Boolean))];
  return [...new Set(bases.flatMap((base) => [base, base.toLowerCase(), base.toUpperCase(), `_${base}`, `_${base.toLowerCase()}`, `__${base}`]).map((base) => `${"https://syjsmvvsvvprxibqoizw.supabase.co/storage/v1/object/public/player-photos/player-photos/"}${base}.png`))];
};
const retryShortlistPhoto = (e, name) => {
  const image = e.currentTarget;
  const candidates = shortlistPhotoCandidates(name);
  const next = Number(image.dataset.photoFallback || 0) + 1;
  if (candidates[next - 1]) { image.dataset.photoFallback = String(next); image.src = candidates[next - 1]; }
  else image.style.display = "none";
};
const defaultTags = [
  { id: "elite", name: "Elite", color: "#62dcff" },
  { id: "high", name: "High Potential", color: "#142b83" },
  { id: "value", name: "Value Potential", color: "#a56bc5" },
  { id: "monitor", name: "Monitor", color: "#ff7416" },
  { id: "contract", name: "Out of contract", color: "#22c55e" },
  { id: "deprioritise", name: "Deprioritise", color: "#f2b807" },
];
const shortlistFolderCacheKey = (userId) => `scoutingShortlistFolders:${userId || "anonymous"}`;
const readShortlistFolderCache = (userId) => {
  try {
    const value = JSON.parse(localStorage.getItem(shortlistFolderCacheKey(userId)) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
};
const writeShortlistFolderCache = (userId, folders) => {
  try {
    localStorage.setItem(shortlistFolderCacheKey(userId), JSON.stringify(folders));
  } catch {
    // Cloud persistence remains the primary store if browser storage is unavailable.
  }
};
export default function ShortlistsPage() {
  const { appState, updateAppState, profile: accountProfile, user } = useAuth();
  const folderCacheUserId = user?.id || "anonymous";
  const folderCacheKey = shortlistFolderCacheKey(folderCacheUserId);
  const [databasePlayers, setDatabasePlayers] = useState([]);
  const [sharedReports, setSharedReports] = useState([]);
  const [sharedLists, setSharedLists] = useState([]), [sharedFolders, setSharedFolders] = useState([]), [showShared, setShowShared] = useState(false), [activeSharedFolderId, setActiveSharedFolderId] = useState("");
  const [scouts, setScouts] = useState([]), [shareEnabled, setShareEnabled] = useState(false), [shareScout, setShareScout] = useState(""), [sharePermission, setSharePermission] = useState("view");
  const [shareShortlistTarget, setShareShortlistTarget] = useState(null), [shareShortlistScout, setShareShortlistScout] = useState(""), [shareShortlistPermission, setShareShortlistPermission] = useState("view"), [sharingShortlist, setSharingShortlist] = useState(false), [shortlistShareMembers, setShortlistShareMembers] = useState([]), [loadingShortlistShares, setLoadingShortlistShares] = useState(false);
  const [shareFolderTarget, setShareFolderTarget] = useState(null), [shareFolderScout, setShareFolderScout] = useState(""), [shareFolderPermission, setShareFolderPermission] = useState("view"), [sharingFolder, setSharingFolder] = useState(false);
  const [folderShareMembers, setFolderShareMembers] = useState([]), [loadingFolderShares, setLoadingFolderShares] = useState(false);
  useEffect(() => {
    if (!appState) return;
    setLists(appState.shortlists || []);
    setTags(appState.tags?.length ? appState.tags : defaultTags);
    if (Array.isArray(appState.shortlistFolders)) {
      setFolders(appState.shortlistFolders);
      writeShortlistFolderCache(folderCacheUserId, appState.shortlistFolders);
    } else {
      const cachedFolders = readShortlistFolderCache(folderCacheUserId);
      setFolders(cachedFolders);
      if (cachedFolders.length) {
        updateAppState({ ...appState, shortlistFolders: cachedFolders }).catch((error) => console.error("Could not sync saved shortlist folders", error));
      }
    }
  }, [appState, folderCacheKey, folderCacheUserId, updateAppState]);
  useEffect(() => {
    if (!accountProfile?.club) return;
    let reportsQuery = supabase.from("club_reports").select("player").eq("status", "Published").eq("club", accountProfile.club);
    if (accountProfile.workspace_type === "independent" && accountProfile.requested_plan === "Independent Scout Free Trial") {
      reportsQuery = reportsQuery.eq("author_id", user?.id);
    }
    reportsQuery.then(({ data }) => setSharedReports(data || []));
  }, [accountProfile?.club, accountProfile?.workspace_type, accountProfile?.requested_plan, user?.id]);
  useEffect(() => { if (accountProfile?.club) supabase.from("profiles").select("id,full_name").eq("club", accountProfile.club).eq("approved", true).then(({ data }) => setScouts((data || []).filter((x) => x.id !== user?.id))); }, [accountProfile?.club, user?.id]);
  useEffect(() => {
    if (!shareFolderTarget || !user) { setFolderShareMembers([]); return; }
    let cancelled = false;
    const loadFolderShares = async () => {
      setLoadingFolderShares(true);
      const folderShareId = `folder-share:${user.id}:${shareFolderTarget.id}`;
      const { data, error } = await supabase.from("shortlist_shares").select("member_id,permission").eq("shortlist_id", folderShareId).eq("owner_id", user.id);
      if (!cancelled) {
        if (error) {
          console.error("Could not load folder sharing details", error);
          setFolderShareMembers([]);
        } else {
          setFolderShareMembers((data || []).map((share) => ({ ...share, name: scouts.find((scout) => String(scout.id) === String(share.member_id))?.full_name || "Scout" })));
        }
        setLoadingFolderShares(false);
      }
    };
    loadFolderShares();
    return () => { cancelled = true; };
  }, [shareFolderTarget, user, scouts]);
  useEffect(() => {
    if (!shareShortlistTarget || !user) { setShortlistShareMembers([]); return; }
    let cancelled = false;
    const loadShares = async () => {
      setLoadingShortlistShares(true);
      const { data, error } = await supabase.from("shortlist_shares").select("member_id,permission").eq("shortlist_id", String(shareShortlistTarget.id)).eq("owner_id", user.id);
      if (cancelled) return;
      if (error) {
        console.error("Could not load shortlist sharing details", error);
        setShortlistShareMembers([]);
      } else {
        setShortlistShareMembers((data || []).map((share) => ({ ...share, name: scouts.find((scout) => String(scout.id) === String(share.member_id))?.full_name || "Scout" })));
      }
      setLoadingShortlistShares(false);
    };
    loadShares();
    return () => { cancelled = true; };
  }, [shareShortlistTarget, user, scouts]);
  useEffect(() => {
    if (!user) return;
    const loadSharedLists = async () => {
      const { data } = await supabase.from("shortlist_shares").select("shortlist_id,owner_id,permission").eq("member_id", user.id);
      const ids = (data || []).map((x) => String(x.shortlist_id));
      if (!ids.length) { setSharedLists([]); setSharedFolders([]); return; }
      const result = await supabase.from("shared_shortlists").select("*").in("shortlist_id", ids);
      const sharedItems = (result.data || []).filter((item) => item.shortlist?.kind !== "squad-plan");
      const mappedItems = sharedItems.map((x) => ({ ...x.shortlist, owner_id: x.owner_id, sharedPermission: data.find((s) => String(s.shortlist_id) === String(x.shortlist_id))?.permission || "view", shared: true }));
      setSharedFolders(mappedItems.filter((item) => item.kind === "shortlist-folder"));
      setSharedLists(mappedItems.filter((item) => item.kind !== "shortlist-folder"));
      const sharedTags = sharedItems.flatMap((x) => x.shortlist?.tags || []);
      if (sharedTags.length) setTags((previous) => [...new Map([...previous, ...sharedTags].map((tag) => [tag.id, tag])).values()]);
    };
    loadSharedLists();
    window.addEventListener("focus", loadSharedLists);
    return () => window.removeEventListener("focus", loadSharedLists);
  }, [user]);
  const [lists, setLists] = useState(() => JSON.parse(localStorage.getItem("scoutingShortlists") || "[]"));
  const [folders, setFolders] = useState(() => appState?.shortlistFolders || readShortlistFolderCache(folderCacheUserId));
  const [activeFolderId, setActiveFolderId] = useState("all");
  const [folderModalOpen, setFolderModalOpen] = useState(false);
  const [folderName, setFolderName] = useState("");
  const [folderColor, setFolderColor] = useState("#3478e8");
  const [folderSaveMessage, setFolderSaveMessage] = useState("");
  const [folderSaving, setFolderSaving] = useState(false);
  const [folderPickerListId, setFolderPickerListId] = useState(null);
  const [createFolderId, setCreateFolderId] = useState(null);
  useEffect(() => {
    if (activeFolderId !== "all" && !folders.some((folder) => String(folder.id) === String(activeFolderId))) setActiveFolderId("all");
  }, [activeFolderId, folders]);
  const currentStorageKey = `scoutingOpenShortlist:${user?.id || "anonymous"}`;
  const currentUserKey = user?.id || "anonymous";
  const readOpenShortlist = (key) => {
    try {
      return JSON.parse(sessionStorage.getItem(key) || "null");
    } catch {
      return null;
    }
  };
  const [current, setCurrent] = useState(() => readOpenShortlist(currentStorageKey));
  const [currentOwnerKey, setCurrentOwnerKey] = useState(currentUserKey);
  useEffect(() => {
    if (currentOwnerKey === currentUserKey) return;
    setCurrentOwnerKey(currentUserKey);
    setCurrent(readOpenShortlist(currentStorageKey));
  }, [currentOwnerKey, currentStorageKey, currentUserKey]);
  useEffect(() => {
    if (currentOwnerKey !== currentUserKey) return;
    try {
      if (current) sessionStorage.setItem(currentStorageKey, JSON.stringify(current));
      else sessionStorage.removeItem(currentStorageKey);
    } catch {
      // Keep the shortlist usable if browser session storage is unavailable.
    }
  }, [current, currentOwnerKey, currentStorageKey, currentUserKey]);
  useEffect(() => {
    if (!user) return;
    const loadOwnedSharedLists = async () => {
      const { data, error } = await supabase.from("shared_shortlists").select("*").eq("owner_id", user.id);
      if (error || !data?.length) return;
      setLists((previous) => previous.map((list) => {
        const snapshot = data.find((item) => String(item.shortlist_id) === String(list.id));
        if (!snapshot) return list;
        const snapshotFolderId = snapshot.shortlist?.sharedFolderId
          ? folders.find((folder) => String(snapshot.shortlist.sharedFolderId).endsWith(`:${folder.id}`))?.id
          : "";
        const next = { ...snapshot.shortlist, folderId: list.folderId || snapshot.shortlist?.folderId || snapshotFolderId || "", owner_id: user.id, shared: true, sharedPermission: list.sharedPermission || "edit" };
        if (current?.id === list.id) setCurrent(next);
        return next;
      }));
    };
    loadOwnedSharedLists();
    window.addEventListener("focus", loadOwnedSharedLists);
    return () => window.removeEventListener("focus", loadOwnedSharedLists);
  }, [user, current?.id, folders]);
  const [show, setShow] = useState(false),
    [name, setName] = useState(""),
    [type, setType] = useState("Formation"),
    [formation, setFormation] = useState("4-3-3"),
    [pick, setPick] = useState(null),
    [search, setSearch] = useState(""),
    [tagPlayer, setTagPlayer] = useState(null),
    [dragging, setDragging] = useState(null),
    [newPlayerOpen, setNewPlayerOpen] = useState(false),
    [newPlayerQuery, setNewPlayerQuery] = useState(""),
    [newPlayers, setNewPlayers] = useState([]),
    [tags, setTags] = useState(() => JSON.parse(localStorage.getItem("scoutingTags") || JSON.stringify(defaultTags))),
    [manageTags, setManageTags] = useState(false),
    [newTagName, setNewTagName] = useState(""),
    [newTagColor, setNewTagColor] = useState("#62dcff");
  useEffect(() => {
    if (!search.trim()) {
      setDatabasePlayers([]);
      return;
    }
    supabase
      .from("players")
      .select("*")
      .ilike("Name", `%${search.trim()}%`)
      .limit(50)
      .then(({ data, error }) => {
        if (error) console.error("Player database error", error);
        if (data) setDatabasePlayers([...new Map(data.map((p) => [String(p.Name || p.name || "").trim().toLowerCase(), p])).values()]);
      });
  }, [search]);
  const published = useMemo(
    () =>
      JSON.parse(localStorage.getItem("scoutingAssignments") || "[]").filter(
        (x) => x.status === "Published",
      ),
    [],
  );
  const playerPool = useMemo(() => {
    const database = databasePlayers.map((p) => ({
      ...p,
      id:
        p.id || p.playerId || p.player_id || p["Player Id"] || p.Name || p.name,
      player:
        p.Name ||
        p.name ||
        [p.firstName || p.first_name, p.lastName || p.last_name]
          .filter(Boolean)
          .join(" "),
      club: p.club || p.team || p.Team,
    }));
    const reports = published.map((x) => ({
      ...x,
      id: x.playerId || x.id,
      player: x.player,
    }));
    return [
      ...new Map(
        [...database, ...reports]
          .filter((p) => p.player)
          .map((p) => [p.id || p.player, p]),
      ).values(),
    ];
  }, [databasePlayers, published]);
  const persist = async (n, sharedUpdate = null) => {
    setLists(n);
    writeShortlistFolderCache(folderCacheUserId, folders);
    updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: n, shortlistFolders: folders, tags });
    if (current?.shared && user && (current.owner_id === user.id || current.sharedPermission === "edit")) {
      const changed = sharedUpdate || n.find((list) => list.id === current.id) || current;
      const sharedPayload = { ...changed };
      delete sharedPayload.shared;
      delete sharedPayload.sharedPermission;
      delete sharedPayload.owner_id;
      sharedPayload.tags = tags;
      const { error } = await supabase
        .from("shared_shortlists")
        .update({ shortlist: sharedPayload, updated_at: new Date().toISOString() })
        .eq("shortlist_id", String(current.id))
        .eq("owner_id", current.owner_id);
      if (error) console.error("Could not save shared shortlist", error);
    }
  };
  const create = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    const folderId = createFolderId !== null ? createFolderId : (activeFolderId !== "all" ? activeFolderId : "");
    const l = { id: Date.now(), name, type, formation, folderId, players: [] };
    const sharedL = shareEnabled && shareScout && user ? { ...l, shared: true, sharedPermission: sharePermission, owner_id: user.id } : l;
    persist([...lists, sharedL]);
    if (shareEnabled && shareScout && user && accountProfile?.club) {
      await supabase.from("shortlist_shares").insert({ shortlist_id: String(l.id), owner_id: user.id, member_id: shareScout, club: accountProfile.club, permission: sharePermission });
      await supabase.from("shared_shortlists").upsert({ shortlist_id: String(l.id), owner_id: user.id, club: accountProfile.club, shortlist: { ...l, tags } }, { onConflict: "shortlist_id" });
    }
    let listToOpen = sharedL;
    if (folderId) {
      try {
        const inheritedShares = await syncListToSharedFolder(l, folderId);
        if (inheritedShares.length) {
          listToOpen = { ...l, shared: true, owner_id: user.id, sharedPermission: inheritedShares[0].permission, sharedFolderId: `folder-share:${user.id}:${folderId}`, sharedFolderName: folders.find((folder) => String(folder.id) === String(folderId))?.name || "Shared folder" };
          const syncedLists = [...lists, listToOpen];
          setLists(syncedLists);
          await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: syncedLists, shortlistFolders: folders, tags });
        }
      } catch (error) { console.error("Could not add the new shortlist to folder shares", error); setFolderSaveMessage("The shortlist was created, but its shared-folder access could not be synced. Reopen Share Folder and share it again."); }
    }
    setName("");
    setCreateFolderId(null);
    setShow(false);
    setShareEnabled(false); setShareScout(""); setSharePermission("view");
    setCurrent(listToOpen);
  };
  const syncListToSharedFolder = async (list, folderId) => {
    if (!user || !accountProfile?.club || !folderId) return [];
    const folder = folders.find((item) => String(item.id) === String(folderId));
    if (!folder) return [];
    const folderShareId = `folder-share:${user.id}:${folder.id}`;
    const { data: shares, error } = await supabase.from("shortlist_shares").select("member_id,permission").eq("shortlist_id", folderShareId).eq("owner_id", user.id);
    if (error) throw error;
    for (const share of shares || []) {
      const snapshot = { ...list, tags, sharedFolderId: folderShareId, sharedFolderName: folder.name };
      delete snapshot.shared;
      delete snapshot.sharedPermission;
      delete snapshot.owner_id;
      const { error: snapshotError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: String(list.id), owner_id: user.id, club: accountProfile.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
      if (snapshotError) throw snapshotError;
      const { data: existing, error: lookupError } = await supabase.from("shortlist_shares").select("shortlist_id").eq("shortlist_id", String(list.id)).eq("member_id", share.member_id).maybeSingle();
      if (lookupError) throw lookupError;
      const payload = { owner_id: user.id, member_id: share.member_id, club: accountProfile.club, permission: share.permission };
      const { error: shareError } = existing
        ? await supabase.from("shortlist_shares").update({ permission: share.permission }).eq("shortlist_id", String(list.id)).eq("member_id", share.member_id)
        : await supabase.from("shortlist_shares").insert({ shortlist_id: String(list.id), ...payload });
      if (shareError) throw shareError;
    }
    return shares || [];
  };
  const saveFolder = async (event) => {
    event.preventDefault();
    const cleanName = folderName.trim();
    if (!cleanName || folderSaving) return;
    const nextFolders = [...folders, { id: `folder-${Date.now()}`, name: cleanName, color: folderColor }];
    setFolderSaving(true);
    setFolders(nextFolders);
    writeShortlistFolderCache(folderCacheUserId, nextFolders);
    setFolderSaveMessage("");
    try {
      await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: lists, shortlistFolders: nextFolders, tags });
    } catch (error) {
      console.error("Could not save shortlist folder to the account", error);
      setFolderSaveMessage("Folder saved on this device. Account sync failed; it will retry when you return to Shortlists.");
    } finally {
      setFolderName("");
      setFolderSaving(false);
      setFolderModalOpen(false);
    }
  };
  const shareFolder = async (event) => {
    event.preventDefault();
    if (!shareFolderTarget || !shareFolderScout || sharingFolder) return;
    setSharingFolder(true);
    try {
      if (!user || !accountProfile?.club) throw new Error("Sharing requires an approved workspace account.");
      const folderShareId = `folder-share:${user.id}:${shareFolderTarget.id}`;
      const shareOne = async (shortlistId, snapshot) => {
        const { error: snapshotError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: shortlistId, owner_id: user.id, club: accountProfile.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
        if (snapshotError) throw snapshotError;
        const { data: existing, error: lookupError } = await supabase.from("shortlist_shares").select("shortlist_id").eq("shortlist_id", shortlistId).eq("member_id", shareFolderScout).maybeSingle();
        if (lookupError) throw lookupError;
        const share = { owner_id: user.id, member_id: shareFolderScout, club: accountProfile.club, permission: shareFolderPermission };
        const { error: shareError } = existing
          ? await supabase.from("shortlist_shares").update({ permission: shareFolderPermission }).eq("shortlist_id", shortlistId).eq("member_id", shareFolderScout)
          : await supabase.from("shortlist_shares").insert({ shortlist_id: shortlistId, ...share });
        if (shareError) throw shareError;
      };
      const marker = { id: folderShareId, kind: "shortlist-folder", name: shareFolderTarget.name, color: shareFolderTarget.color || "#3478e8", ownerName: accountProfile.full_name || "A scout" };
      await shareOne(folderShareId, marker);
      const folderLists = lists.filter((list) => String(list.folderId || "") === String(shareFolderTarget.id));
      for (const list of folderLists) {
        // Keep the owner's folderId in their saved shortlist, while sharedFolderId groups the copy for recipients.
        const snapshot = { ...list, tags, sharedFolderId: folderShareId, sharedFolderName: shareFolderTarget.name };
        delete snapshot.shared;
        delete snapshot.sharedPermission;
        delete snapshot.owner_id;
        await shareOne(String(list.id), snapshot);
      }
      const nextLists = lists.map((list) => String(list.folderId || "") === String(shareFolderTarget.id)
        ? { ...list, shared: true, owner_id: user.id, sharedPermission: shareFolderPermission, sharedFolderId: folderShareId, sharedFolderName: shareFolderTarget.name }
        : list);
      setLists(nextLists);
      await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: folders, tags });
      if (current && folderLists.some((list) => String(list.id) === String(current.id))) {
        const sharedCurrent = nextLists.find((list) => String(list.id) === String(current.id));
        if (sharedCurrent) setCurrent(sharedCurrent);
      }
      setFolderSaveMessage(`“${shareFolderTarget.name}” and ${folderLists.length} shortlist${folderLists.length === 1 ? "" : "s"} shared successfully.`);
      setShareFolderTarget(null);
      setShareFolderScout("");
      setShareFolderPermission("view");
      const { data: refreshedShares } = await supabase.from("shortlist_shares").select("member_id,permission").eq("shortlist_id", folderShareId).eq("owner_id", user.id);
      setFolderShareMembers((refreshedShares || []).map((share) => ({ ...share, name: scouts.find((scout) => String(scout.id) === String(share.member_id))?.full_name || "Scout" })));
    } catch (error) {
      console.error("Could not share shortlist folder", error);
      setFolderSaveMessage(`Could not share the folder: ${error.message || "please try again."}`);
    } finally {
      setSharingFolder(false);
    }
  };
  const stopSharingFolder = async (memberId) => {
    if (!shareFolderTarget || !user || sharingFolder) return;
    const scoutName = folderShareMembers.find((member) => String(member.member_id) === String(memberId))?.name || "this scout";
    if (!window.confirm(`Stop sharing “${shareFolderTarget.name}” and its shortlists with ${scoutName}?`)) return;
    setSharingFolder(true);
    try {
      const folderShareId = `folder-share:${user.id}:${shareFolderTarget.id}`;
      const folderLists = lists.filter((list) => String(list.folderId || "") === String(shareFolderTarget.id));
      const ids = [folderShareId, ...folderLists.map((list) => String(list.id))];
      const { error: deleteError } = await supabase.from("shortlist_shares").delete().eq("owner_id", user.id).eq("member_id", memberId).in("shortlist_id", ids);
      if (deleteError) throw deleteError;
      for (const id of ids) {
        const { data: remaining, error: remainingError } = await supabase.from("shortlist_shares").select("shortlist_id").eq("shortlist_id", id).eq("owner_id", user.id).limit(1);
        if (remainingError) throw remainingError;
        if (!remaining?.length) {
          const { error: snapshotError } = await supabase.from("shared_shortlists").delete().eq("shortlist_id", id).eq("owner_id", user.id);
          if (snapshotError) throw snapshotError;
        }
      }
      setFolderShareMembers((previous) => previous.filter((member) => String(member.member_id) !== String(memberId)));
      setFolderSaveMessage(`Stopped sharing “${shareFolderTarget.name}” with ${scoutName}.`);
    } catch (error) {
      console.error("Could not stop sharing shortlist folder", error);
      setFolderSaveMessage(`Could not stop sharing the folder: ${error.message || "please try again."}`);
    } finally {
      setSharingFolder(false);
    }
  };
  const removeListFromSharedFolder = async (list, folderId) => {
    if (!user || !folderId || !list.sharedFolderId) return list;
    const folderShareId = `folder-share:${user.id}:${folderId}`;
    if (String(list.sharedFolderId) !== folderShareId) return list;
    const { data: folderShares, error: folderError } = await supabase.from("shortlist_shares").select("member_id").eq("shortlist_id", folderShareId).eq("owner_id", user.id);
    if (folderError) throw folderError;
    const inheritedMemberIds = (folderShares || []).map((share) => share.member_id);
    if (inheritedMemberIds.length) {
      const { error: removeError } = await supabase.from("shortlist_shares").delete().eq("shortlist_id", String(list.id)).eq("owner_id", user.id).in("member_id", inheritedMemberIds);
      if (removeError) throw removeError;
    }
    const { data: remaining, error: remainingError } = await supabase.from("shortlist_shares").select("member_id").eq("shortlist_id", String(list.id)).eq("owner_id", user.id);
    if (remainingError) throw remainingError;
    const { sharedFolderId, sharedFolderName, shared, sharedPermission, owner_id, ...unfoldered } = list;
    if (remaining?.length) {
      const snapshot = { ...unfoldered, tags };
      const { error: snapshotError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: String(list.id), owner_id: user.id, club: accountProfile?.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
      if (snapshotError) throw snapshotError;
      return { ...unfoldered, shared: true, owner_id: user.id, sharedPermission: remaining[0].permission || "view" };
    }
    const { error: snapshotError } = await supabase.from("shared_shortlists").delete().eq("shortlist_id", String(list.id)).eq("owner_id", user.id);
    if (snapshotError) throw snapshotError;
    return unfoldered;
  };
  const moveShortlist = async (listId, folderId) => {
    const original = lists.find((list) => String(list.id) === String(listId));
    if (!original) return;
    let moved = { ...original, folderId: folderId || "" };
    try {
      if (original.folderId && String(original.folderId) !== String(folderId || "")) {
        moved = await removeListFromSharedFolder(moved, original.folderId);
        moved.folderId = folderId || "";
      }
      const inheritedShares = folderId ? await syncListToSharedFolder(moved, folderId) : [];
      if (inheritedShares.length) {
        moved = { ...moved, shared: true, owner_id: user.id, sharedPermission: inheritedShares[0].permission, sharedFolderId: `folder-share:${user.id}:${folderId}`, sharedFolderName: folders.find((folder) => String(folder.id) === String(folderId))?.name || "Shared folder" };
      }
      const nextLists = lists.map((list) => String(list.id) === String(listId) ? moved : list);
      setLists(nextLists);
      await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: folders, tags });
      if (current?.id === original.id) setCurrent(moved);
      if (!folderId && activeFolderId === original.folderId) setActiveFolderId("all");
      if (original.folderId && !folderId) setFolderSaveMessage(`“${original.name}” moved back to unfiled shortlists.`);
    } catch (error) {
      console.error("Could not move shortlist", error);
      setFolderSaveMessage(`The shortlist could not be moved: ${error.message || "please try again."}`);
    }
  };
  const renameFolder = (folder) => {
    const nextName = window.prompt("Rename folder", folder.name)?.trim();
    if (!nextName || nextName === folder.name) return;
    const nextFolders = folders.map((item) => item.id === folder.id ? { ...item, name: nextName } : item);
    setFolders(nextFolders);
    writeShortlistFolderCache(folderCacheUserId, nextFolders);
    updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: lists, shortlistFolders: nextFolders, tags }).catch((error) => { console.error("Could not save renamed shortlist folder", error); setFolderSaveMessage("Folder change saved on this device, but account sync failed."); });
  };
  const deleteFolder = (folder) => {
    const count = lists.filter((list) => String(list.folderId) === String(folder.id)).length;
    if (!window.confirm(`Delete “${folder.name}”? ${count ? `${count} shortlist${count === 1 ? "" : "s"} will be moved to Unfiled.` : ""}`)) return;
    const nextFolders = folders.filter((item) => item.id !== folder.id);
    const nextLists = lists.map((list) => String(list.folderId) === String(folder.id) ? { ...list, folderId: "" } : list);
    setFolders(nextFolders);
    setLists(nextLists);
    writeShortlistFolderCache(folderCacheUserId, nextFolders);
    if (activeFolderId === folder.id) setActiveFolderId("all");
    updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: nextFolders, tags }).catch((error) => { console.error("Could not save folder deletion", error); setFolderSaveMessage("Folder change saved on this device, but account sync failed."); });
  };
  const renameShortlist = async (list) => {
    const nextName = window.prompt("Rename shortlist", list.name)?.trim();
    if (!nextName || nextName === list.name) return;
    const renamed = { ...list, name: nextName };
    const nextLists = lists.map((item) => String(item.id) === String(list.id) ? renamed : item);
    setLists(nextLists);
    if (current?.id === list.id) setCurrent(renamed);
    try {
      await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: folders, tags });
      if (user) {
        const { data: shares, error } = await supabase.from("shortlist_shares").select("member_id").eq("shortlist_id", String(list.id)).eq("owner_id", user.id).limit(1);
        if (error) throw error;
        if (shares?.length) {
          const snapshot = { ...renamed, tags };
          delete snapshot.shared;
          delete snapshot.sharedPermission;
          delete snapshot.owner_id;
          const { error: saveError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: String(list.id), owner_id: user.id, club: accountProfile?.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
          if (saveError) throw saveError;
        }
      }
    } catch (error) {
      console.error("Could not save renamed shortlist", error);
      setFolderSaveMessage("The shortlist name changed on this device, but account or shared access did not sync.");
    }
  };
  const shareShortlist = async (event) => {
    event.preventDefault();
    if (!shareShortlistTarget || !shareShortlistScout || !user || !accountProfile?.club || sharingShortlist) return;
    setSharingShortlist(true);
    try {
      const shortlistId = String(shareShortlistTarget.id);
      const snapshot = { ...shareShortlistTarget, tags };
      delete snapshot.shared;
      delete snapshot.sharedPermission;
      delete snapshot.owner_id;
      const { error: snapshotError } = await supabase.from("shared_shortlists").upsert({ shortlist_id: shortlistId, owner_id: user.id, club: accountProfile.club, shortlist: snapshot, updated_at: new Date().toISOString() }, { onConflict: "shortlist_id" });
      if (snapshotError) throw snapshotError;
      const { data: existing, error: lookupError } = await supabase.from("shortlist_shares").select("shortlist_id").eq("shortlist_id", shortlistId).eq("member_id", shareShortlistScout).maybeSingle();
      if (lookupError) throw lookupError;
      const payload = { owner_id: user.id, member_id: shareShortlistScout, club: accountProfile.club, permission: shareShortlistPermission };
      const { error: shareError } = existing
        ? await supabase.from("shortlist_shares").update({ permission: shareShortlistPermission }).eq("shortlist_id", shortlistId).eq("member_id", shareShortlistScout)
        : await supabase.from("shortlist_shares").insert({ shortlist_id: shortlistId, ...payload });
      if (shareError) throw shareError;
      const nextLists = lists.map((list) => String(list.id) === shortlistId ? { ...list, shared: true, owner_id: user.id, sharedPermission: shareShortlistPermission } : list);
      setLists(nextLists);
      await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: folders, tags });
      setShortlistShareMembers((previous) => {
        const scout = scouts.find((item) => String(item.id) === String(shareShortlistScout));
        const nextMember = { member_id: shareShortlistScout, permission: shareShortlistPermission, name: scout?.full_name || "Scout" };
        return [...previous.filter((item) => String(item.member_id) !== String(shareShortlistScout)), nextMember];
      });
      setShareShortlistScout("");
      setFolderSaveMessage(`“${shareShortlistTarget.name}” shared successfully.`);
    } catch (error) {
      console.error("Could not share shortlist", error);
      setFolderSaveMessage(`Could not share the shortlist: ${error.message || "please try again."}`);
    } finally {
      setSharingShortlist(false);
    }
  };
  const stopSharingShortlist = async (memberId) => {
    if (!shareShortlistTarget || !user || sharingShortlist) return;
    const member = shortlistShareMembers.find((item) => String(item.member_id) === String(memberId));
    const name = member?.name || "this scout";
    if (!window.confirm(`Stop sharing “${shareShortlistTarget.name}” with ${name}?`)) return;
    setSharingShortlist(true);
    try {
      const shortlistId = String(shareShortlistTarget.id);
      const { error } = await supabase.from("shortlist_shares").delete().eq("shortlist_id", shortlistId).eq("owner_id", user.id).eq("member_id", memberId);
      if (error) throw error;
      const { data: remaining, error: remainingError } = await supabase.from("shortlist_shares").select("member_id").eq("shortlist_id", shortlistId).eq("owner_id", user.id).limit(1);
      if (remainingError) throw remainingError;
      if (!remaining?.length) {
        const { error: snapshotError } = await supabase.from("shared_shortlists").delete().eq("shortlist_id", shortlistId).eq("owner_id", user.id);
        if (snapshotError) throw snapshotError;
        const nextLists = lists.map((list) => String(list.id) === shortlistId ? ((list.sharedFolderId) ? list : (({ shared, sharedPermission, owner_id, ...localList }) => localList)(list)) : list);
        setLists(nextLists);
        if (String(current?.id) === shortlistId) setCurrent(nextLists.find((list) => String(list.id) === shortlistId) || null);
        await updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: nextLists, shortlistFolders: folders, tags });
      }
      setShortlistShareMembers((previous) => previous.filter((item) => String(item.member_id) !== String(memberId)));
      setFolderSaveMessage(`Stopped sharing “${shareShortlistTarget.name}” with ${name}.`);
    } catch (error) {
      console.error("Could not stop sharing shortlist", error);
      setFolderSaveMessage(`Could not stop sharing the shortlist: ${error.message || "please try again."}`);
    } finally {
      setSharingShortlist(false);
    }
  };
  const add = (p, pos) => {
    if (current.players.some((x) => x.id === p.id && x.slot === pos)) return;
    const n = {
      ...current,
      players: [...current.players, { ...p, slot: pos }],
    };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
    setPick(null);
    setSearch("");
  };
  const remove = (id, pos) => {
    const n = {
      ...current,
      players: current.players.filter((x) => !(x.id === id && x.slot === pos)),
    };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
  };
  const savePositionLabel = (pos, value) => {
    const positionLabels = { ...(current.positionLabels || {}), [pos]: value.trim() };
    const n = { ...current, positionLabels };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
  };
  const toggleTag = (tagId) => {
    const n = {
      ...tagPlayer,
      tags: tagPlayer.tags?.includes(tagId)
        ? tagPlayer.tags.filter((x) => x !== tagId)
        : [...(tagPlayer.tags || []), tagId],
    };
    const updated = {
      ...current,
      players: current.players.map((x) =>
        x.id === tagPlayer.id && x.slot === tagPlayer.slot ? n : x,
      ),
    };
    persist(lists.map((x) => (x.id === current.id ? updated : x)), updated);
    setCurrent(updated);
    setTagPlayer(n);
  };
  const createTag = (e) => {
    e.preventDefault();
    if (!newTagName.trim()) return;
    const n = {
      id: `custom-${Date.now()}`,
      name: newTagName.trim(),
      color: newTagColor,
    };
    const next = [...tags, n];
    setTags(next);
    updateAppState({ ...appState, assignments: appState?.assignments || [], shortlists: appState?.shortlists || [], shortlistFolders: folders, tags: next });
    if (current?.shared && user && (current.owner_id === user.id || current.sharedPermission === "edit")) {
      supabase.from("shared_shortlists").update({ shortlist: { ...current, tags: next, shared: undefined, sharedPermission: undefined, owner_id: undefined }, updated_at: new Date().toISOString() }).eq("shortlist_id", String(current.id)).eq("owner_id", current.owner_id).then(({ error }) => { if (error) console.error("Could not save shared shortlist tags", error); });
    }
    setNewTagName("");
  };
  const changeFormation = (value) => {
    const slots = [...(formationRows[value] || [])].flatMap((row, i) =>
      row.map((pos, j) => `${pos}-${j}`),
    );
    const counters = {};
    let fallbackCursor = 0;
    const players = current.players.map((p) => {
      const label = p.slot.split("-")[0];
      const matches = slots.filter((s) => s.split("-")[0] === label);
      const index = counters[label] || 0;
      counters[label] = index + 1;
      const slot = matches[index] || matches[matches.length - 1] || slots[fallbackCursor++ % slots.length] || "CF-0";
      return { ...p, slot };
    });
    const n = { ...current, formation: value, players };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
  };
  const addSelectedPlayers = () => {
    if (!newPlayers.length) return;
    const fallback =
      (formationRows[current.formation] || flexRows).flat()[0] + "-0";
    const players = [
      ...current.players,
      ...newPlayers.map((p) => ({ ...p, slot: p.addSlot || fallback })),
    ];
    const n = { ...current, players };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
    setNewPlayers([]);
    setNewPlayerQuery("");
    setNewPlayerOpen(false);
  };
  const reorder = (pos, targetId, clientY, targetRect) => {
    if (!dragging || dragging.pos !== pos || dragging.id === targetId) return;
    const inPos = current.players.filter((x) => x.slot === pos);
    const from = inPos.findIndex((x) => x.id === dragging.id);
    const targetIndex = inPos.findIndex((x) => x.id === targetId);
    if (from < 0 || targetIndex < 0) return;
    const droppedBelow = targetRect && clientY > targetRect.top + targetRect.height / 2;
    let to = targetIndex + (droppedBelow ? 1 : 0);
    if (from < to) to -= 1;
    if (from === to) { setDragging(null); return; }
    const ordered = [...inPos];
    const [moved] = ordered.splice(from, 1);
    ordered.splice(to, 0, moved);
    const n = {
      ...current,
      players: [...current.players.filter((x) => x.slot !== pos), ...ordered],
    };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
    setDragging(null);
  };
  const moveToPosition = (targetPos) => {
    if (!dragging || dragging.pos === targetPos) return;
    const moved = current.players.find((p) => p.id === dragging.id && p.slot === dragging.pos);
    if (!moved) return;
    const n = { ...current, players: current.players.map((p) => p === moved ? { ...p, slot: targetPos } : p) };
    persist(lists.map((x) => (x.id === current.id ? n : x)), n);
    setCurrent(n);
    setDragging(null);
  };
  const tagManager = manageTags && (
    <div className="sr-modal" onClick={() => setManageTags(false)}>
      <form
        className="sr-form sr-tag-modal"
        onClick={(e) => e.stopPropagation()}
        onSubmit={createTag}
      >
        <div className="sr-form-head">
          <div>
            <div className="sr-kicker">SHORTLIST TAGS</div>
            <h2>Manage Tags</h2>
          </div>
          <button
            type="button"
            className="sr-close"
            onClick={() => setManageTags(false)}
          >
            ×
          </button>
        </div>
        <label className="sr-field">
          <span>New tag</span>
          <input
            value={newTagName}
            onChange={(e) => setNewTagName(e.target.value)}
            placeholder="Tag name"
          />
        </label>
        <div className="sr-color-palette">
          {[
            "#62dcff",
            "#142b83",
            "#a56bc5",
            "#ff7416",
            "#22c55e",
            "#f2b807",
            "#ef4444",
            "#3b82f6",
            "#ec4899",
            "#14b8a6",
            "#64748b",
            "#84cc16",
          ].map((c) => (
            <button
              type="button"
              key={c}
              style={{ background: c }}
              className={newTagColor === c ? "selected" : ""}
              onClick={() => setNewTagColor(c)}
            />
          ))}
        </div>
        <button className="sr-cyan">Add Tag</button>
        <div className="sr-existing-tags">
          {tags.map((t) => (
            <span key={t.id} style={{ background: t.color }}>
              {t.name}
            </span>
          ))}
        </div>
      </form>
    </div>
  );
  const tagModal = tagPlayer && (
    <div className="sr-modal" onClick={() => setTagPlayer(null)}>
      <section
        className="sr-form sr-tag-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sr-form-head">
          <div>
            <div className="sr-kicker">PLAYER TAGS</div>
            <h2>{tagPlayer.player}</h2>
          </div>
          <button className="sr-close" onClick={() => setTagPlayer(null)}>
            ×
          </button>
        </div>
        <p>Click a tag to add or remove it.</p>
        <div className="sr-tag-list">
          {tags.map((t) => (
            <button
              key={t.id}
              style={{
                background: t.color,
                color: t.color === "#f2b807" ? "#071d3d" : "#fff",
              }}
              className={tagPlayer.tags?.includes(t.id) ? "active" : ""}
              onClick={() => toggleTag(t.id)}
            >
              {t.name}
            </button>
          ))}
        </div>
        <button className="sr-cyan" onClick={() => setTagPlayer(null)}>
          Done
        </button>
      </section>
    </div>
  );
  const zone = (pos) => {
    const players = current.players.filter((x) => x.slot === pos),
      matches = playerPool.filter((x) =>
        x.player.toLowerCase().includes(search.toLowerCase()),
      );
    return (
      <div className={`sr-zone sr-zone-position-${pos.split("-")[0].toLowerCase()}`} key={pos} onDragOver={(e) => e.preventDefault()} onDrop={() => moveToPosition(pos)}>
        <div className="sr-zone-head">
          <input
            className="sr-position-label-input"
            aria-label="Edit position label"
            title="Edit position label"
            value={current.positionLabels?.[pos] ?? pos.split("-")[0]}
            onChange={(e) => setCurrent({
              ...current,
              positionLabels: { ...(current.positionLabels || {}), [pos]: e.target.value },
            })}
            onBlur={(e) => savePositionLabel(pos, e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
          />
          <span>{players.length} Players</span>
          <button
            onClick={() => {
              setPick(pick === pos ? null : pos);
              setSearch("");
            }}
          >
            +
          </button>
        </div>
        <div className="sr-zone-list">
          {players.map((p) => (
            <div
              className="sr-pitch-player-card"
              key={p.id}
              draggable
              onDragStart={(e) => {
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/plain", String(p.id));
                setDragging({ id: p.id, pos });
              }}
              onDragEnd={() => setDragging(null)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (dragging?.pos === pos) {
                  reorder(pos, p.id, e.clientY, e.currentTarget.getBoundingClientRect());
                } else {
                  moveToPosition(pos);
                }
              }}
              style={{
                "--tag-color":
                  tags.find((t) => (p.tags || []).includes(t.id))?.color ||
                  "#f7fbff",
              }}
            >
              <button
                onClick={() => {
                  localStorage.setItem("scoutingProfilePlayer", p.player);
                  localStorage.setItem("scoutingProfileOrigin", "shortlists");
                  window.location.href = "/scouting-reports";
                }}
              >
                <img
                  className="sr-shortlist-player-photo"
                  src={p.Photo || p.photo || p._photoUrl || shortlistPhoto(p.player)}
                  alt=""
                  onError={(e) => retryShortlistPhoto(e, p.player)}
                />
                <span>
                  <strong>{p.player}</strong>
                  <small><ClubName club={p.club || p.Club || p.team || p.Team} size={16} /></small>
                  <em className="sr-report-count">
                    ▤{" "}
                    {
                      (appState?.assignments || []).filter(
                        (x) =>
                        x.player === p.player && x.status === "Published",
                      ).length
                      + sharedReports.filter((x) => x.player === p.player).length
                    }
                  </em>
                </span>
              </button>
              <div
                className="sr-player-highlight"
                style={{
                  background:
                    tags.find((t) => (p.tags || []).includes(t.id))?.color ||
                    "transparent",
                }}
              />
              <div className="sr-card-tags">
                {(p.tags || []).map((id) => {
                  const t = tags.find((x) => x.id === id);
                  return t ? (
                    <i
                      key={id}
                      title={t.name}
                      style={{ background: t.color }}
                    />
                  ) : null;
                })}
              </div>
              <button
                className="sr-player-tag-button"
                title="Add or remove tags"
                onClick={(e) => {
                  e.stopPropagation();
                  setTagPlayer(p);
                }}
              >
                🏷
              </button>
              <button onClick={() => remove(p.id, pos)}>×</button>
            </div>
          ))}
        </div>
        {pick === pos && (
          <div className="sr-zone-picker">
            <input
              autoFocus
              placeholder="Search player..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            {search.trim() &&
              matches.map((p) => (
                <button key={p.id} onClick={() => add(p, pos)}>
                  {p.player}
                  <small><ClubName club={p.club || p.Club || p.team || p.Team} size={16} /></small>
                </button>
              ))}
          </div>
        )}
      </div>
    );
  };
  const del = (id) => {
    if (window.confirm("Delete this shortlist?")) {
      persist(lists.filter((x) => x.id !== id));
      if (user) { supabase.from("shortlist_shares").delete().eq("shortlist_id", String(id)).eq("owner_id", user.id); supabase.from("shared_shortlists").delete().eq("shortlist_id", String(id)).eq("owner_id", user.id); }
      if (String(current?.id) === String(id)) setCurrent(null);
    }
  };
  const newPlayerMatches = playerPool.filter((p) =>
    p.player.toLowerCase().includes(newPlayerQuery.toLowerCase()),
  );
  const newPlayerModal = newPlayerOpen && (
    <div className="sr-modal" onClick={() => setNewPlayerOpen(false)}>
      <section
        className="sr-form sr-new-player-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sr-form-head">
          <h2>Add New Players</h2>
          <button className="sr-close" onClick={() => setNewPlayerOpen(false)}>
            ×
          </button>
        </div>
        <input
          className="sr-search"
          autoFocus
          placeholder="Search player..."
          value={newPlayerQuery}
          onChange={(e) => {
            setNewPlayerQuery(e.target.value);
            setSearch(e.target.value);
          }}
        />
        <div className="sr-new-player-results">
          {newPlayerMatches.map((p) => {
            const chosen = newPlayers.find((x) => x.id === p.id);
            return (
              <div
                className={chosen ? "selected" : ""}
                key={p.id}
                onClick={() =>
                  setNewPlayers(
                    chosen
                      ? newPlayers.filter((x) => x.id !== p.id)
                      : [
                          ...newPlayers,
                          {
                            ...p,
                            addSlot: (
                              formationRows[current.formation] || flexRows
                            )
                              .flat()
                              .map((x, i) => x + "-" + i)[0],
                          },
                        ],
                  )
                }
              >
                <img src={shortlistPhoto(p.player)} alt="" />
                <span>
                  <strong>{p.player}</strong>
                  <small><ClubName club={p.club || p.Club || p.team || p.Team} size={16} /></small>
                </span>
                {chosen && (
                  <select
                    value={chosen.addSlot}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) =>
                      setNewPlayers(
                        newPlayers.map((x) =>
                          x.id === p.id ? { ...x, addSlot: e.target.value } : x,
                        ),
                      )
                    }
                  >
                    {(formationRows[current.formation] || flexRows)
                      .flat()
                      .map((x, i) => (
                        <option key={x + "-" + i} value={x + "-" + i}>
                          {x}
                        </option>
                      ))}
                  </select>
                )}
              </div>
            );
          })}
        </div>
        <div className="sr-actions">
          <button
            className="sr-outline"
            onClick={() => setNewPlayerOpen(false)}
          >
            Cancel
          </button>
          <button className="sr-cyan" onClick={addSelectedPlayers}>
            Add
          </button>
        </div>
      </section>
    </div>
  );
  if (current)
    return (
      <main className="sr-page sr-shortlist-view sr-shortlists-page">
        <button className="sr-back" onClick={() => setCurrent(null)}>
          ‹ Back to Your Shortlists
        </button>
        <section className="sr-dashboard-head">
          <div>
            <div className="sr-kicker">SHORTLIST</div>
            <h1>{current.name}</h1>
            {current.type === "Flex" ? (
              <p>Flex · 14 positions</p>
            ) : (
              <select
                className="sr-formation-select"
                value={current.formation}
                onChange={(e) => changeFormation(e.target.value)}
              >
                {formations.map((f) => (
                  <option key={f}>{f}</option>
                ))}
              </select>
            )}
          </div>
          <div className="sr-shortlist-head-actions">
            <button className="sr-delete-list" onClick={() => del(current.id)}>
              Delete Shortlist
            </button>
            <div className="sr-shortlist-actions">
              <button
                className="sr-outline"
                onClick={() => setManageTags(true)}
              >
                Add Tags
              </button>
              <button
                className="sr-outline"
                onClick={() => setPick("new-player")}
              >
                Add New Player
              </button>
            </div>
          </div>
        </section>
        <div className="sr-tag-legend">
          {tags.map((t) => (
            <span key={t.id}>
              <i style={{ background: t.color }} />
              {t.name}
            </span>
          ))}
        </div>
        <div className="sr-real-pitch">
          <div className="sr-goal-box top" />
          {(current.type === "Flex"
            ? flexRows
            : formationRows[current.formation] || formationRows["4-3-3"]
          ).map((r, i) => (
            <div
              className={`sr-pitch-row row-${i} row-size-${r.length}`}
              style={{
                zIndex: r.some((pos) => pick && pick.startsWith(pos + "-"))
                  ? 10
                  : 2,
              }}
              key={i}
            >
              {r.map((pos, j) => zone(pos + "-" + j))}
            </div>
          ))}
          <div className="sr-centre-circle" />
          <div className="sr-halfway-line" />
          <div className="sr-goal-box bottom" />
        </div>
        {tagModal}
        {tagManager}
        {newPlayerModal}
      </main>
    );
  const openShortlist = (list) => {
    const rows = list.type === "Flex" ? flexRows : (formationRows[list.formation] || formationRows["4-3-3"]);
    const positions = rows.flatMap((row) => row.map((position, index) => `${position}-${index}`));
    let fallbackCursor = 0;
    const players = (list.players || []).map((player) => {
      const role = String(player.slot || "").split("-")[0];
      if (positions.includes(player.slot)) return player;
      const fallbackRole = positions.find((slot) => slot.split("-")[0] === role) || positions[fallbackCursor++ % positions.length] || "CF-0";
      return { ...player, slot: fallbackRole };
    });
    setCurrent({ ...list, players });
  };
  const activeFolder = folders.find((folder) => String(folder.id) === String(activeFolderId));
  const activeSharedFolder = sharedFolders.find((folder) => String(folder.id) === String(activeSharedFolderId));
  const visibleShortlists = showShared
    ? activeSharedFolder
      ? sharedLists.filter((list) => String(list.sharedFolderId || "") === String(activeSharedFolder.id))
      : sharedLists.filter((list) => !list.sharedFolderId)
    : activeFolder
      ? lists.filter((list) => String(list.folderId || "") === String(activeFolder.id))
      : activeFolderId === "all"
        ? lists.filter((list) => !list.folderId)
        : [];
  const folderCard = (folder) => {
    const count = lists.filter((list) => String(list.folderId) === String(folder.id)).length;
    return <article className="sr-list-card sr-shortlist-folder-card" key={folder.id} style={{ "--folder-color": folder.color || "#3478e8" }}>
      <button type="button" className="sr-folder-open" onClick={() => setActiveFolderId(folder.id)}>
        <span className="sr-folder-icon" aria-hidden="true"><svg viewBox="0 0 48 48" focusable="false"><path d="M5 12a4 4 0 0 1 4-4h10l5 5h15a4 4 0 0 1 4 4v18a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4V12Z" /></svg></span>
        <h2>{folder.name}</h2>
        <p>{count} shortlist{count === 1 ? "" : "s"}</p>
        <span className="sr-folder-open-label">Open ›</span>
      </button>
      <div className="sr-folder-actions">
        <button type="button" aria-label={`Share ${folder.name}`} title="Share folder" onClick={() => { setShareFolderTarget(folder); setShareFolderScout(""); }}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><path d="m8.7 10.6 6.6-4.1M8.7 13.4l6.6 4.1" /></svg></button>
        <button type="button" aria-label={`Rename ${folder.name}`} title="Rename folder" onClick={() => renameFolder(folder)}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 20h9" /><path d="m16.5 3.5 4 4L8 20l-5 1 1-5Z" /></svg></button>
        <button type="button" aria-label={`Delete ${folder.name}`} title="Delete folder" onClick={() => deleteFolder(folder)}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6m4 4v6m6-6v6" /></svg></button>
      </div>
    </article>;
  };
  const sharedFolderCard = (folder) => <article className="sr-list-card sr-shortlist-folder-card" key={folder.id} style={{ "--folder-color": folder.color || "#3478e8" }}><button type="button" className="sr-folder-open" onClick={() => setActiveSharedFolderId(folder.id)}><span className="sr-folder-icon" aria-hidden="true"><svg viewBox="0 0 48 48" focusable="false"><path d="M5 12a4 4 0 0 1 4-4h10l5 5h15a4 4 0 0 1 4 4v18a4 4 0 0 1-4 4H9a4 4 0 0 1-4-4V12Z" /></svg></span><h2>{folder.name}</h2><p>Shared folder{folder.ownerName ? ` · ${folder.ownerName}` : ""}</p><span className="sr-folder-open-label">Open ›</span></button></article>;
  const shortlistCard = (list) => <article className="sr-list-card sr-shortlist-item" key={list.id}>
    <button type="button" className="sr-list-card-button" onClick={() => openShortlist(list)}>
      <h2>{list.name}</h2>
      <p>{list.type === "Flex" ? "Flex formation" : list.formation}</p>
      <span>{(list.players || []).length} players</span>
    </button>
    {!showShared && <div className="sr-shortlist-card-actions">
      {folderPickerListId === list.id ? <label className="sr-shortlist-folder-select"><span>Move to</span><select autoFocus aria-label={`Move ${list.name} to folder`} value={list.folderId || "__choose__"} onChange={(event) => { moveShortlist(list.id, event.target.value); setFolderPickerListId(null); }}><option value="__choose__" disabled>Choose a folder</option><option value="">Unfiled</option>{folders.map((folder) => <option key={folder.id} value={folder.id}>{folder.name}</option>)}</select><button type="button" aria-label="Close folder selector" onClick={() => setFolderPickerListId(null)}>Cancel</button></label> : <>
        <button type="button" className="sr-shortlist-icon-button" aria-label={`Share ${list.name}`} title="Share shortlist" onClick={() => { setShareShortlistTarget(list); setShareShortlistScout(""); }}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" /><path d="m8.7 10.6 6.6-4.1M8.7 13.4l6.6 4.1" /></svg></button>
        {list.folderId ? <button type="button" className="sr-shortlist-icon-button" aria-label={`Remove ${list.name} from folder`} title="Remove from folder" onClick={() => moveShortlist(list.id, "")}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /><path d="M3.5 10h17M12 13H7m0 0 2.5-2.5M7 13l2.5 2.5" /></svg></button> : <button type="button" className="sr-shortlist-icon-button" aria-label={`Move ${list.name} to a folder`} title={folders.length ? "Move to folder" : "Create a folder first"} disabled={!folders.length} onClick={() => setFolderPickerListId(list.id)}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 6.5A2.5 2.5 0 0 1 5.5 4H10l2 2h6.5A2.5 2.5 0 0 1 21 8.5v9a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" /><path d="M3.5 9h17" /></svg></button>}
        <button type="button" className="sr-shortlist-icon-button" aria-label={`Rename ${list.name}`} title="Rename shortlist" onClick={() => renameShortlist(list)}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M12 20h9" /><path d="m16.5 3.5 4 4L8 20l-5 1 1-5Z" /></svg></button>
        <button type="button" className="sr-shortlist-icon-button" aria-label={`Delete ${list.name}`} title="Delete shortlist" onClick={() => del(list.id)}><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6m4 4v6m6-6v6" /></svg></button>
      </>}
    </div>}
  </article>;
  return (
    <main className="sr-page sr-shortlists-page">
      <section className="sr-dashboard-head">
        <div>
          <div className="sr-kicker">SCOUTING PLATFORM</div>
          <h1>Your Shortlists</h1>
          <p>{showShared ? "Shortlists and folders other scouts have shared with you." : activeFolder ? `Shortlists inside ${activeFolder.name}.` : "Group shortlists into folders for each team, squad or recruitment project."}</p>
        </div>
        <div className="sr-head-actions"><button className={!showShared ? "sr-cyan" : "sr-outline"} onClick={() => { setShowShared(false); setActiveFolderId("all"); setActiveSharedFolderId(""); }}>My Shortlists</button><button className={showShared ? "sr-cyan" : "sr-outline"} onClick={() => { setShowShared(true); setActiveFolderId("all"); setActiveSharedFolderId(""); }}>Shared Shortlists</button>{!showShared && <><button className="sr-outline" onClick={() => setFolderModalOpen(true)}>Create Folder</button><button className="sr-cyan" onClick={() => { setCreateFolderId(activeFolder?.id || ""); setShow(true); }}>Create New Shortlist</button></>}</div>
      </section>
      {folderSaveMessage && <p className="sr-folder-save-message" role="status">{folderSaveMessage}</p>}
      {!showShared && activeFolder && <button type="button" className="sr-back sr-shortlist-folder-back" onClick={() => setActiveFolderId("all")}>‹ All folders and shortlists</button>}
      {showShared && activeSharedFolder && <button type="button" className="sr-back sr-shortlist-folder-back" onClick={() => setActiveSharedFolderId("")}>‹ All shared folders and shortlists</button>}
      {!showShared && activeFolderId === "all" ? <section className="sr-shortlist-folder-section">
        <div className="sr-shortlist-section-heading"><h2>Folders &amp; unfiled shortlists</h2></div>
        {folders.length || visibleShortlists.length ? <div className="sr-list-grid sr-shortlist-folder-grid">{folders.map(folderCard)}{visibleShortlists.map(shortlistCard)}</div> : <div className="sr-empty">Create a folder for a team or project, or create a shortlist on its own.</div>}
      </section> : <section className="sr-shortlist-folder-section">
        <div className="sr-shortlist-section-heading"><h2>{showShared ? activeSharedFolder?.name || "Shared shortlists" : activeFolder?.name}</h2>{(showShared && activeSharedFolder) && <span>{visibleShortlists.length} shortlist{visibleShortlists.length === 1 ? "" : "s"}</span>}{!showShared && activeFolder && <span>{visibleShortlists.length} shortlist{visibleShortlists.length === 1 ? "" : "s"}</span>}</div>
        {showShared && !activeSharedFolder ? (sharedFolders.length || visibleShortlists.length ? <div className="sr-list-grid sr-shortlist-folder-grid">{sharedFolders.map(sharedFolderCard)}{visibleShortlists.map(shortlistCard)}</div> : <div className="sr-empty">No shortlists or folders have been shared with you yet.</div>) : visibleShortlists.length > 0 ? <div className="sr-list-grid sr-shortlist-list-grid">{visibleShortlists.map(shortlistCard)}</div> : <div className="sr-empty">{showShared ? "This shared folder has no shortlists yet." : "This folder is empty. Create a shortlist here or move one into it."}</div>}
      </section>}
      {folderModalOpen && <div className="sr-modal sr-shortlist-folder-modal" onClick={() => !folderSaving && setFolderModalOpen(false)}><form className="sr-form sr-shortlist-folder-form" onClick={(event) => event.stopPropagation()} onSubmit={saveFolder}><div className="sr-form-head"><div><div className="sr-kicker">SHORTLIST ORGANISATION</div><h2>Create Folder</h2></div><button type="button" className="sr-close" disabled={folderSaving} onClick={() => setFolderModalOpen(false)}>×</button></div><label className="sr-field"><span>Folder name</span><input autoFocus value={folderName} onChange={(event) => setFolderName(event.target.value)} placeholder="For example, Preston North End" /></label><label className="sr-field sr-folder-colour-field"><span>Folder colour</span><div className="sr-folder-colour-picker"><input aria-label="Choose folder colour" type="color" value={folderColor} onChange={(event) => setFolderColor(event.target.value)} /><output>{folderColor.toUpperCase()}</output><span className="sr-folder-colour-preview" style={{ background: folderColor }} /></div></label><div className="sr-actions"><button type="button" className="sr-outline" disabled={folderSaving} onClick={() => setFolderModalOpen(false)}>Cancel</button><button className="sr-cyan" disabled={!folderName.trim() || folderSaving}>{folderSaving ? "Saving…" : "Create Folder"}</button></div></form></div>}
      {shareFolderTarget && <div className="sr-modal" onClick={() => !sharingFolder && setShareFolderTarget(null)}><form className="sr-form" onClick={(event) => event.stopPropagation()} onSubmit={shareFolder}><div className="sr-form-head"><div><div className="sr-kicker">SHORTLIST SHARING</div><h2>Share “{shareFolderTarget.name}”</h2></div><button type="button" className="sr-close" disabled={sharingFolder} onClick={() => setShareFolderTarget(null)}>×</button></div><p>Sharing this folder also shares every shortlist inside it. New shortlists added to the folder will be shared automatically.</p>
        <div className="sr-folder-sharing-list"><h3>People with access</h3>{loadingFolderShares ? <p>Loading…</p> : folderShareMembers.length ? folderShareMembers.map((member) => <div className="sr-folder-sharee" key={member.member_id}><div><strong>{member.name}</strong><span>{member.permission === "edit" ? "Can view and edit" : "View only"}</span></div><button type="button" className="sr-folder-stop-sharing" disabled={sharingFolder} onClick={() => stopSharingFolder(member.member_id)}>Stop sharing</button></div>) : <p>This folder isn’t shared with anyone yet.</p>}</div>
        <label className="sr-field"><span>Share with scout</span><select autoFocus value={shareFolderScout} onChange={(event) => setShareFolderScout(event.target.value)}><option value="">Select scout</option>{scouts.map((scout) => <option key={scout.id} value={scout.id}>{scout.full_name || scout.id}</option>)}</select></label><label className="sr-field"><span>Permission</span><select value={shareFolderPermission} onChange={(event) => setShareFolderPermission(event.target.value)}><option value="view">View only</option><option value="edit">View and edit</option></select></label><div className="sr-actions"><button type="button" className="sr-outline" disabled={sharingFolder} onClick={() => setShareFolderTarget(null)}>Done</button><button className="sr-cyan" disabled={!shareFolderScout || sharingFolder}>{sharingFolder ? "Sharing…" : "Share Folder"}</button></div></form></div>}
      {shareShortlistTarget && <div className="sr-modal" onClick={() => !sharingShortlist && setShareShortlistTarget(null)}><form className="sr-form" onClick={(event) => event.stopPropagation()} onSubmit={shareShortlist}><div className="sr-form-head"><div><div className="sr-kicker">SHORTLIST SHARING</div><h2>Share “{shareShortlistTarget.name}”</h2></div><button type="button" className="sr-close" disabled={sharingShortlist} onClick={() => setShareShortlistTarget(null)}>×</button></div><p>Share this shortlist, including its players and tags, with another scout.</p>
        <div className="sr-folder-sharing-list"><h3>People with access</h3>{loadingShortlistShares ? <p>Loading…</p> : shortlistShareMembers.length ? shortlistShareMembers.map((member) => <div className="sr-folder-sharee" key={member.member_id}><div><strong>{member.name}</strong><span>{member.permission === "edit" ? "Can view and edit" : "View only"}</span></div><button type="button" className="sr-folder-stop-sharing" disabled={sharingShortlist} onClick={() => stopSharingShortlist(member.member_id)}>Stop sharing</button></div>) : <p>This shortlist isn’t shared with anyone yet.</p>}</div>
        <label className="sr-field"><span>Share with scout</span><select autoFocus value={shareShortlistScout} onChange={(event) => setShareShortlistScout(event.target.value)}><option value="">Select scout</option>{scouts.map((scout) => <option key={scout.id} value={scout.id}>{scout.full_name || scout.id}</option>)}</select></label><label className="sr-field"><span>Permission</span><select value={shareShortlistPermission} onChange={(event) => setShareShortlistPermission(event.target.value)}><option value="view">View only</option><option value="edit">View and edit</option></select></label><div className="sr-actions"><button type="button" className="sr-outline" disabled={sharingShortlist} onClick={() => setShareShortlistTarget(null)}>Done</button><button className="sr-cyan" disabled={!shareShortlistScout || sharingShortlist}>{sharingShortlist ? "Sharing…" : "Share Shortlist"}</button></div></form></div>}
      {show && (
        <div className="sr-modal">
          <form className="sr-form sr-shortlist-create" onSubmit={create}>
            <div className="sr-form-head">
              <div>
                <div className="sr-kicker">SHORTLIST MANAGEMENT</div>
                <h2>Create Shortlist</h2>
              </div>
              <button
                type="button"
                className="sr-close"
                onClick={() => setShow(false)}
              >
                ×
              </button>
            </div>
            <div className="sr-choice-grid sr-choice-small">
              <button
                type="button"
                className={type === "Flex" ? "active" : ""}
                onClick={() => setType("Flex")}
              >
                <h2>Flex</h2>
                <p>All 14 positions.</p>
              </button>
              <button
                type="button"
                className={type === "Formation" ? "active" : ""}
                onClick={() => setType("Formation")}
              >
                <h2>Formation</h2>
                <p>Regular 11 positions.</p>
                {type === "Formation" && (
                  <select
                    value={formation}
                    onChange={(e) => setFormation(e.target.value)}
                  >
                    {formations.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                )}
              </button>
            </div>
            <label className="sr-field">
              <span>Shortlist Name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Shortlist name"
              />
            </label>
            <label className="sr-field"><span>Folder</span><select value={createFolderId !== null ? createFolderId : (activeFolder?.id || "")} onChange={(event) => setCreateFolderId(event.target.value)}><option value="">Unfiled</option>{folders.map((folder) => <option key={folder.id} value={folder.id}>{folder.name}</option>)}</select></label>
            <label className="sr-share-check"><input type="checkbox" checked={shareEnabled} onChange={(e) => setShareEnabled(e.target.checked)} /> Share shortlist with another scout</label>
            {shareEnabled && <div className="sr-form-grid"><label className="sr-field"><span>Scout</span><select value={shareScout} onChange={(e) => setShareScout(e.target.value)}><option value="">Select scout</option>{scouts.map((s) => <option key={s.id} value={s.id}>{s.full_name || s.id}</option>)}</select></label><label className="sr-field"><span>Permission</span><select value={sharePermission} onChange={(e) => setSharePermission(e.target.value)}><option value="view">View only</option><option value="edit">View and edit</option></select></label></div>}
            <div className="sr-actions">
              <button
                type="button"
                className="sr-outline"
                onClick={() => setShow(false)}
              >
                Cancel
              </button>
              <button className="sr-cyan">Save</button>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}
