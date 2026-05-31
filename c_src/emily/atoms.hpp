// Cached atom terms.
//
// fine::Atom objects defined at namespace scope register themselves on
// construction and have their terms created once in the NIF load callback
// (fine's load runs fine::__private__::init_atoms — see deps/fine).
// Encoding such an atom (via fine::encode) then returns the prebuilt term
// directly (Encoder<Atom> uses the cached `term`), with no per-call
// atom-table lookup.
//
// Defining the reply and dtype atoms here — rather than building them at
// point of use with enif_make_atom / fine::Atom("...") — keeps every atom
// on the hot async-reply path (notably `:ok`, posted for every successful
// op) off the lookup.

#pragma once

#include <fine.hpp>

namespace emily::atoms {

// Async reply atoms (see emily/async.hpp, emily_nif.cpp).
inline auto ok = fine::Atom("ok");
inline auto error = fine::Atom("error");
inline auto stopped = fine::Atom("stopped");
inline auto argument = fine::Atom("argument");
inline auto runtime = fine::Atom("runtime");
inline auto unknown = fine::Atom("unknown");

// Nx dtype "kind" atoms (see emily/dtype.hpp).
inline auto f = fine::Atom("f");
inline auto bf = fine::Atom("bf");
inline auto s = fine::Atom("s");
inline auto u = fine::Atom("u");
inline auto c = fine::Atom("c");
inline auto pred = fine::Atom("pred");

} // namespace emily::atoms
