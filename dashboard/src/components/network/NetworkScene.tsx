'use client';

// tactical 3D cyber warfare command canvas
// features holographic grid plane, sweeping radar scanner, hardware server models,
// fiber-optic packet conduits, threat breach lasers, and tactical host HUD dossier

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useAppStore } from '@/store';
import { NETWORK_LINKS, NODE_POSITIONS, STATUS_COLORS, getNode } from '@/engine/network';
import { NetworkFallback2D } from './NetworkFallback2D';

export function NetworkScene() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const topology = useAppStore((s) => s.topology);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const alerts = useAppStore((s) => s.alerts);
  const viewMode = useAppStore((s) => s.viewMode);

  const [hasWebGl, setHasWebGl] = useState(true);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const topologyRef = useRef(topology);
  topologyRef.current = topology;

  const alertsRef = useRef(alerts);
  alertsRef.current = alerts;

  const selectedRef = useRef(selectedNodeId);
  selectedRef.current = selectedNodeId;

  const selectedNode = selectedNodeId
    ? getNode({ nodes: topology, blockedIps: new Set(), honeypotActive: false, isolatedNodes: new Set() }, selectedNodeId)
    : null;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || viewMode === '2d') return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: 'high-performance',
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(container.clientWidth, container.clientHeight);
    } catch {
      setHasWebGl(false);
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#05070c');
    scene.fog = new THREE.FogExp2('#05070c', 0.035);

    const camera = new THREE.PerspectiveCamera(
      42,
      container.clientWidth / container.clientHeight,
      0.1,
      100,
    );
    camera.position.set(0, 9.5, 11);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI / 2 - 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 24;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.25;

    // ambient & tactical directional lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xd0e0ff, 0.85);
    mainLight.position.set(6, 15, 8);
    scene.add(mainLight);

    const cyanPoint = new THREE.PointLight(0x00f0ff, 1.4, 25);
    cyanPoint.position.set(0, 4, 0);
    scene.add(cyanPoint);

    const amberPoint = new THREE.PointLight(0xffb703, 1.0, 20);
    amberPoint.position.set(-4, 6, -3);
    scene.add(amberPoint);

    // root container group
    const root = new THREE.Group();
    root.position.set(0, -0.6, 0);
    scene.add(root);

    // 1. HOLOGRAPHIC CYBER GRID & SECTOR FLOOR
    const gridHelper = new THREE.GridHelper(26, 32, 0x00f0ff, 0x121a2c);
    gridHelper.position.y = -0.05;
    root.add(gridHelper);

    // glowing boundary rings around sectors
    const createSectorRing = (x: number, z: number, radius: number, color: number) => {
      const ringGeom = new THREE.RingGeometry(radius - 0.06, radius, 48);
      const ringMat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.35 });
      const ring = new THREE.Mesh(ringGeom, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.01, z);
      root.add(ring);
    };
    createSectorRing(1.2, -3, 3.2, 0x00f0ff);   // DMZ & Perimeter
    createSectorRing(1, 0, 4.8, 0x3a86ff);      // Core Directory & Services
    createSectorRing(1, 3, 5.0, 0xffb703);      // Endpoint Fleet

    // 2. SWEEPING RADAR SCAN LINE
    const radarGroup = new THREE.Group();
    const radarConeGeom = new THREE.ConeGeometry(8, 0.02, 32, 1, true, 0, Math.PI / 4);
    const radarMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
    });
    const radarCone = new THREE.Mesh(radarConeGeom, radarMat);
    radarCone.rotation.x = Math.PI / 2;
    radarCone.position.y = 0.02;
    radarGroup.add(radarCone);
    root.add(radarGroup);

    // 3. HARDWARE-ACCURATE 3D SERVER & HARDWARE MODELS
    interface NodeObject {
      nodeId: string;
      group: THREE.Group;
      coreMesh: THREE.Mesh;
      accentMeshes: THREE.Mesh[];
      shieldMesh: THREE.Mesh;
      breachBeams: THREE.Group;
    }
    const nodeObjects: NodeObject[] = [];
    const interactiveMeshes: THREE.Mesh[] = [];

    topologyRef.current.forEach((node) => {
      const pos = NODE_POSITIONS[node.nodeId] || { x: 0, y: 0, z: 0 };
      const statusColor = STATUS_COLORS[node.status] || '#06d6a0';
      const nodeGroup = new THREE.Group();
      nodeGroup.position.set(pos.x, 0, pos.z);

      const accentMeshes: THREE.Mesh[] = [];

      // Base pedestal for every server/terminal
      const pedestalGeom = new THREE.CylinderGeometry(0.55, 0.65, 0.12, 6);
      const pedestalMat = new THREE.MeshStandardMaterial({
        color: 0x111622,
        roughness: 0.6,
        metalness: 0.8,
      });
      const pedestal = new THREE.Mesh(pedestalGeom, pedestalMat);
      pedestal.position.y = 0.06;
      nodeGroup.add(pedestal);

      let coreMesh: THREE.Mesh;

      if (node.type === 'firewall') {
        // Bastion Gateway Tower with concentric energy rings
        const towerGeom = new THREE.BoxGeometry(0.55, 1.1, 0.55);
        const towerMat = new THREE.MeshStandardMaterial({
          color: 0x1c2436,
          metalness: 0.85,
          roughness: 0.25,
        });
        coreMesh = new THREE.Mesh(towerGeom, towerMat);
        coreMesh.position.y = 0.65;

        // Energy shield emitter ring
        const ringGeom = new THREE.TorusGeometry(0.48, 0.04, 12, 24);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.8 });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.9;
        nodeGroup.add(ring);
        accentMeshes.push(ring);

        // Warning beacon tip
        const beaconGeom = new THREE.OctahedronGeometry(0.18);
        const beaconMat = new THREE.MeshBasicMaterial({ color: 0xffb703 });
        const beacon = new THREE.Mesh(beaconGeom, beaconMat);
        beacon.position.y = 1.35;
        nodeGroup.add(beacon);
        accentMeshes.push(beacon);
      } else if (node.type === 'domain_controller') {
        // High-density enterprise dual rack chassis
        const rackGeom = new THREE.BoxGeometry(0.85, 1.3, 0.7);
        const rackMat = new THREE.MeshStandardMaterial({
          color: 0x151c2a,
          metalness: 0.9,
          roughness: 0.2,
        });
        coreMesh = new THREE.Mesh(rackGeom, rackMat);
        coreMesh.position.y = 0.75;

        // Blinking LED status strips
        for (let r = 0; r < 4; r++) {
          const ledGeom = new THREE.BoxGeometry(0.65, 0.04, 0.02);
          const ledMat = new THREE.MeshBasicMaterial({ color: r === 0 ? 0x00f0ff : 0x06d6a0 });
          const led = new THREE.Mesh(ledGeom, ledMat);
          led.position.set(0, 0.4 + r * 0.22, 0.36);
          nodeGroup.add(led);
          accentMeshes.push(led);
        }
      } else if (node.type === 'database') {
        // Multi-tier storage database silo
        const siloGeom = new THREE.CylinderGeometry(0.48, 0.48, 1.0, 24);
        const siloMat = new THREE.MeshStandardMaterial({
          color: 0x182030,
          metalness: 0.8,
          roughness: 0.3,
        });
        coreMesh = new THREE.Mesh(siloGeom, siloMat);
        coreMesh.position.y = 0.6;

        // Animated rotating magnetic disc rings
        for (let d = 0; d < 3; d++) {
          const discGeom = new THREE.TorusGeometry(0.52, 0.03, 8, 24);
          const discMat = new THREE.MeshBasicMaterial({ color: 0x3a86ff });
          const disc = new THREE.Mesh(discGeom, discMat);
          disc.rotation.x = Math.PI / 2;
          disc.position.y = 0.35 + d * 0.25;
          nodeGroup.add(disc);
          accentMeshes.push(disc);
        }
      } else if (node.type === 'honeypot') {
        // Crystalline deception trap node
        const trapGeom = new THREE.IcosahedronGeometry(0.45, 0);
        const trapMat = new THREE.MeshStandardMaterial({
          color: 0x221832,
          metalness: 0.9,
          roughness: 0.1,
          wireframe: false,
        });
        coreMesh = new THREE.Mesh(trapGeom, trapMat);
        coreMesh.position.y = 0.7;

        // Strobe hazard ring
        const hazardGeom = new THREE.TorusGeometry(0.55, 0.05, 12, 16);
        const hazardMat = new THREE.MeshBasicMaterial({ color: 0xffb703, wireframe: true });
        const hazardRing = new THREE.Mesh(hazardGeom, hazardMat);
        hazardRing.rotation.x = Math.PI / 2;
        hazardRing.position.y = 0.7;
        nodeGroup.add(hazardRing);
        accentMeshes.push(hazardRing);
      } else if (node.type === 'workstation') {
        // Hexagonal endpoint workstation console
        const deskGeom = new THREE.CylinderGeometry(0.42, 0.48, 0.55, 6);
        const deskMat = new THREE.MeshStandardMaterial({ color: 0x161e2c, metalness: 0.7, roughness: 0.4 });
        coreMesh = new THREE.Mesh(deskGeom, deskMat);
        coreMesh.position.y = 0.35;

        // Holographic screen projection
        const screenGeom = new THREE.PlaneGeometry(0.38, 0.26);
        const screenMat = new THREE.MeshBasicMaterial({
          color: 0x00f0ff,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        const screen = new THREE.Mesh(screenGeom, screenMat);
        screen.rotation.x = -Math.PI / 6;
        screen.position.set(0, 0.78, 0.1);
        nodeGroup.add(screen);
        accentMeshes.push(screen);
      } else {
        // Standard server blade enclosure (web, app, mail, backup)
        const srvGeom = new THREE.BoxGeometry(0.65, 0.9, 0.55);
        const srvMat = new THREE.MeshStandardMaterial({
          color: 0x172030,
          metalness: 0.8,
          roughness: 0.3,
        });
        coreMesh = new THREE.Mesh(srvGeom, srvMat);
        coreMesh.position.y = 0.55;

        // Optical port grid
        const portGeom = new THREE.BoxGeometry(0.45, 0.12, 0.02);
        const portMat = new THREE.MeshBasicMaterial({ color: 0x06d6a0 });
        const port = new THREE.Mesh(portGeom, portMat);
        port.position.set(0, 0.75, 0.28);
        nodeGroup.add(port);
        accentMeshes.push(port);
      }

      coreMesh.userData = { nodeId: node.nodeId };
      nodeGroup.add(coreMesh);
      interactiveMeshes.push(coreMesh);

      // Crystalline force field shield for isolated hosts
      const shieldGeom = new THREE.IcosahedronGeometry(1.0, 1);
      const shieldMat = new THREE.MeshBasicMaterial({
        color: 0x3a86ff,
        wireframe: true,
        transparent: true,
        opacity: 0.4,
      });
      const shieldMesh = new THREE.Mesh(shieldGeom, shieldMat);
      shieldMesh.position.y = 0.65;
      shieldMesh.visible = node.status === 'isolated';
      nodeGroup.add(shieldMesh);

      // Active attack breach beam container
      const breachBeams = new THREE.Group();
      breachBeams.visible = false;
      const breachRingGeom = new THREE.RingGeometry(0.75, 0.85, 32);
      const breachRingMat = new THREE.MeshBasicMaterial({ color: 0xff2a55, side: THREE.DoubleSide });
      const breachRing = new THREE.Mesh(breachRingGeom, breachRingMat);
      breachRing.rotation.x = Math.PI / 2;
      breachRing.position.y = 0.05;
      breachBeams.add(breachRing);
      nodeGroup.add(breachBeams);

      root.add(nodeGroup);

      nodeObjects.push({
        nodeId: node.nodeId,
        group: nodeGroup,
        coreMesh,
        accentMeshes,
        shieldMesh,
        breachBeams,
      });
    });

    // 4. FIBER-OPTIC DATA CONDUITS & LIGHT PIPELINES
    const conduitItems: Array<{ line: THREE.Line; src: string; dst: string }> = [];
    NETWORK_LINKS.forEach(([src, dst]) => {
      const p1 = NODE_POSITIONS[src] || { x: 0, y: 0, z: 0 };
      const p2 = NODE_POSITIONS[dst] || { x: 0, y: 0, z: 0 };

      const points = [
        new THREE.Vector3(p1.x, 0.15, p1.z),
        new THREE.Vector3(p2.x, 0.15, p2.z),
      ];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({
        color: 0x1a2942,
        transparent: true,
        opacity: 0.65,
        linewidth: 1.5,
      });
      const line = new THREE.Line(geom, mat);
      root.add(line);
      conduitItems.push({ line, src, dst });
    });

    // 5. HIGH-SPEED TRAVELING DATA PACKET STREAM
    const packetCount = 28;
    const packetGeom = new THREE.SphereGeometry(0.08, 8, 8);
    const packetMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    const packetInstMesh = new THREE.InstancedMesh(packetGeom, packetMat, packetCount);
    root.add(packetInstMesh);

    // 6. THREAT BREACH LASERS (Red attack lines when under breach)
    const laserCount = 12;
    const laserGeom = new THREE.SphereGeometry(0.12, 8, 8);
    const laserMat = new THREE.MeshBasicMaterial({ color: 0xff2a55 });
    const laserInstMesh = new THREE.InstancedMesh(laserGeom, laserMat, laserCount);
    root.add(laserInstMesh);

    const dummy = new THREE.Object3D();

    // RAYCASTING FOR INTERACTION
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const handlePointerDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactiveMeshes);

      if (intersects.length > 0) {
        const clickedId = intersects[0].object.userData.nodeId;
        setSelectedNodeId(clickedId === selectedRef.current ? null : clickedId);
      } else {
        setSelectedNodeId(null);
      }
    };

    const handlePointerMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(interactiveMeshes);

      if (intersects.length > 0) {
        setHoveredNode(intersects[0].object.userData.nodeId);
        canvas.style.cursor = 'pointer';
      } else {
        setHoveredNode(null);
        canvas.style.cursor = 'grab';
      }
    };

    canvas.addEventListener('click', handlePointerDown);
    canvas.addEventListener('mousemove', handlePointerMove);

    // 60FPS RENDER & ANIMATION LOOP
    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);

      const elapsed = clock.getElapsedTime();
      controls.update();

      // Rotate radar scanner
      radarGroup.rotation.y = elapsed * 0.9;

      const curTopology = topologyRef.current;
      const curAlerts = alertsRef.current;
      const curSelected = selectedRef.current;

      const activeThreatNodes = curAlerts
        .filter((a) => (a.status === 'new' || a.status === 'investigating') && !a.isFalsePositive)
        .map((a) => a.targetNodeId);

      // Update node states, glowing materials, and breach indicators
      nodeObjects.forEach((item) => {
        const node = curTopology.find((n) => n.nodeId === item.nodeId);
        if (!node) return;

        const isCompromised = node.status === 'compromised' || node.status === 'encrypted';
        const isAttacked = activeThreatNodes.includes(item.nodeId);
        const isSelected = curSelected === item.nodeId;

        // Core mesh material
        const mat = item.coreMesh.material as THREE.MeshStandardMaterial;
        const baseHex = STATUS_COLORS[node.status] || '#06d6a0';

        if (isCompromised) {
          const pulse = 1 + Math.sin(elapsed * 6) * 0.12;
          item.group.scale.set(pulse, pulse, pulse);
          mat.emissive.set('#ff2a55');
          mat.emissiveIntensity = 0.85;
        } else if (isSelected) {
          item.group.scale.set(1.15, 1.15, 1.15);
          mat.emissive.set('#00f0ff');
          mat.emissiveIntensity = 0.5;
        } else {
          item.group.scale.set(1, 1, 1);
          mat.emissive.set(baseHex);
          mat.emissiveIntensity = 0.15;
        }

        // Shield status
        item.shieldMesh.visible = node.status === 'isolated';
        if (item.shieldMesh.visible) {
          item.shieldMesh.rotation.y = elapsed * 0.5;
        }

        // Breach beams
        item.breachBeams.visible = isAttacked;
        if (isAttacked) {
          item.breachBeams.rotation.z = elapsed * 2;
        }
      });

      // Update conduit colors (turn fiery red along attack corridors)
      conduitItems.forEach(({ line, src, dst }) => {
        const isThreatPath = activeThreatNodes.includes(src) || activeThreatNodes.includes(dst);
        const mat = line.material as THREE.LineBasicMaterial;
        if (isThreatPath) {
          mat.color.set('#ff2a55');
          mat.opacity = 0.95;
        } else {
          mat.color.set('#1a2942');
          mat.opacity = 0.5;
        }
      });

      // Animate normal data packets traveling along links
      for (let i = 0; i < packetCount; i++) {
        const link = NETWORK_LINKS[i % NETWORK_LINKS.length];
        const p1 = NODE_POSITIONS[link[0]] || { x: 0, y: 0, z: 0 };
        const p2 = NODE_POSITIONS[link[1]] || { x: 0, y: 0, z: 0 };

        const progress = (elapsed * 0.6 + i * (1 / packetCount)) % 1;
        const px = THREE.MathUtils.lerp(p1.x, p2.x, progress);
        const py = 0.15 + Math.sin(progress * Math.PI) * 0.15;
        const pz = THREE.MathUtils.lerp(p1.z, p2.z, progress);

        dummy.position.set(px, py, pz);
        dummy.scale.set(0.7, 0.7, 0.7);
        dummy.updateMatrix();
        packetInstMesh.setMatrixAt(i, dummy.matrix);
      }
      packetInstMesh.instanceMatrix.needsUpdate = true;

      // Animate hostile breach lasers if active threats exist
      if (activeThreatNodes.length > 0) {
        laserInstMesh.visible = true;
        for (let j = 0; j < laserCount; j++) {
          const targetNodeId = activeThreatNodes[j % activeThreatNodes.length];
          const targetPos = NODE_POSITIONS[targetNodeId] || { x: 0, y: 0, z: 0 };
          const entryPos = { x: 0, y: 0, z: -3 }; // Perimeter firewall entry

          const progress = (elapsed * 1.5 + j * (1 / laserCount)) % 1;
          const lx = THREE.MathUtils.lerp(entryPos.x, targetPos.x, progress);
          const ly = 0.4 + Math.sin(progress * Math.PI) * 0.4;
          const lz = THREE.MathUtils.lerp(entryPos.z, targetPos.z, progress);

          dummy.position.set(lx, ly, lz);
          dummy.scale.set(1.2, 1.2, 1.2);
          dummy.updateMatrix();
          laserInstMesh.setMatrixAt(j, dummy.matrix);
        }
        laserInstMesh.instanceMatrix.needsUpdate = true;
      } else {
        laserInstMesh.visible = false;
      }

      renderer.render(scene, camera);
    };

    animate();

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      canvas.removeEventListener('click', handlePointerDown);
      canvas.removeEventListener('mousemove', handlePointerMove);
      renderer.dispose();
      controls.dispose();
    };
  }, [viewMode, setSelectedNodeId]);

  if (viewMode === '2d' || !hasWebGl) {
    return <NetworkFallback2D />;
  }

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        background: '#05070c',
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />

      {/* TOP TACTICAL HUD OVERLAY */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 14,
          display: 'flex',
          gap: 12,
          pointerEvents: 'none',
        }}
      >
        <div
          style={{
            background: 'rgba(10, 14, 23, 0.85)',
            border: '1px solid var(--border-bright)',
            padding: '4px 8px',
            borderRadius: 2,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--cyan)', boxShadow: '0 0 6px var(--cyan)' }} />
          <span>RADAR: SCANNING ENTERPRISE GRID</span>
        </div>
      </div>

      {/* HOVER BADGE */}
      {hoveredNode && (
        <div
          style={{
            position: 'absolute',
            top: 12,
            right: 14,
            background: 'rgba(10, 14, 23, 0.92)',
            border: '1px solid var(--cyan)',
            padding: '4px 10px',
            borderRadius: 2,
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--cyan)',
            pointerEvents: 'none',
            boxShadow: '0 0 10px rgba(0, 240, 255, 0.25)',
          }}
        >
          TARGET ACQUIRED: {hoveredNode.toUpperCase()} (CLICK TO INSPECT)
        </div>
      )}

      {/* SELECTED NODE DOSSIER CARD */}
      {selectedNode && (
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(10, 14, 23, 0.95)',
            border: '1px solid var(--border-bright)',
            padding: '14px 18px',
            borderRadius: 3,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            zIndex: 10,
            width: 320,
            boxShadow: '0 12px 36px rgba(0,0,0,0.8)',
            borderLeft: `3px solid ${STATUS_COLORS[selectedNode.status]}`,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontWeight: 800, color: '#fff', fontSize: 13, letterSpacing: '0.04em' }}>
              {selectedNode.hostname.toUpperCase()}
            </span>
            <span
              style={{
                color: STATUS_COLORS[selectedNode.status],
                fontWeight: 700,
                fontSize: 10,
                padding: '1px 6px',
                background: 'rgba(0,0,0,0.4)',
                border: `1px solid ${STATUS_COLORS[selectedNode.status]}`,
                borderRadius: 2,
              }}
            >
              {selectedNode.status.toUpperCase()}
            </span>
          </div>

          <div style={{ color: 'var(--text-2)', fontSize: 10.5, marginBottom: 4 }}>
            IP: <span style={{ color: '#fff' }}>{selectedNode.ip}</span> • OS: {selectedNode.os}
          </div>
          <div style={{ color: 'var(--text-2)', fontSize: 10.5, marginBottom: 4 }}>
            SERVICES: {selectedNode.services.join(', ') || 'NONE'}
          </div>
          <div style={{ color: 'var(--text-2)', fontSize: 10.5, marginBottom: 6 }}>
            OPEN PORTS: {selectedNode.openPorts.map((p) => `:${p}`).join(' ') || 'NONE'}
          </div>

          {selectedNode.vulnerabilities.length > 0 ? (
            <div style={{ padding: '6px 8px', background: 'rgba(255, 42, 85, 0.15)', border: '1px solid var(--red)', borderRadius: 2, color: 'var(--red)', fontSize: 10 }}>
              ⚠️ EXPLOITABLE CVE: {selectedNode.vulnerabilities.join(', ')}
            </div>
          ) : (
            <div style={{ color: 'var(--text-3)', fontSize: 10 }}>NO KNOWN CVE EXPLOITS ACTIVE</div>
          )}

          <div style={{ display: 'flex', gap: 6, marginTop: 10 }}>
            <button
              onClick={() => setSelectedNodeId(null)}
              style={{
                flex: 1,
                padding: '4px',
                background: '#161e2e',
                border: '1px solid var(--border)',
                color: 'var(--text-2)',
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                cursor: 'pointer',
              }}
            >
              CLOSE DOSSIER
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
